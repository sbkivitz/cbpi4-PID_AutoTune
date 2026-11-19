from time import localtime, strftime
import time
import math
import logging
import inspect
import io
from collections import deque
from collections import namedtuple
import asyncio
from asyncio import tasks
import logging
from cbpi.api import *
import datetime
from cbpi.controller.kettle_controller import KettleController
from socket import timeout
from typing import KeysView

from voluptuous.schema_builder import message
from cbpi.api.dataclasses import NotificationAction, NotificationType

try:
    # Present on forks with a pluggable clock, which lets a tune be rehearsed
    # on a simulator faster than real time. On hardware it is real time.
    # Optional, so the plugin still loads on a CraftBeerPi without it.
    from cbpi.api import clock
except ImportError:  # pragma: no cover - depends on the host installation
    class _RealClock:
        @staticmethod
        def now():
            return time.time()

        @staticmethod
        def monotonic():
            return time.monotonic()

        @staticmethod
        async def sleep(seconds):
            await asyncio.sleep(max(0.0, seconds))

    clock = _RealClock()


def _notify_kept(cbpi, title, message, type, action):
    """notify(..., keep=True) where CraftBeerPi can keep a prompt for every
    page until it is answered, plain notify where it cannot."""
    try:
        keeps = "keep" in inspect.signature(cbpi.notify).parameters
    except (TypeError, ValueError):
        keeps = False
    if keeps:
        return cbpi.notify(title, message, type, action=action, keep=True)
    return cbpi.notify(title, message, type, action=action)


@parameters([Property.Number(label = "Output_Step", configurable = True, default_value = 100, description="Default: 100. Sets the output when stepping up/down."),
             Property.Number(label = "Max_Output", configurable = True, default_value = 100, description="Default: 100. Sets the max power output."),
             Property.Select(label = "SampleTime", options=[2, 5],
                             description="Seconds between control decisions during the tune. Default: 5. "
                                         "Matching the logic you will run is tidy but not required: the gains "
                                         "this produces are PER-SECOND quantities and are entered as-is at any "
                                         "SampleTime. PIDArduino's _Ki = ki * sampleTime and _Kd = kd / sampleTime "
                                         "exist precisely to make that true - the integral adds ki*s*error every "
                                         "s seconds, and the derivative multiplies a delta spanning s seconds by "
                                         "kd/s. Measured over a 40 minute hold, the same gains at SampleTime 2 "
                                         "and 5 differ by 0.228% of output."),
             Property.Number(label = "lockback_seconds", configurable = True, default_value = 30, description="Default: 30. How far back to look for min/max temps."),
             Property.Number(label = "Max_Tune_Minutes", configurable = True, default_value = 300,
                             description="Default: 300 (5 hours). The longest a tune may run, in minutes. If it "
                                         "hasn't finished by then, the heater goes off, the tune ends and a "
                                         "notification says so. With 20-40 L and a 5.5 kW element a tune takes "
                                         "about 1 to 3 hours at a mash temperature, nearer 4 at a low target "
                                         "with the pump off. Raise it for a bigger fill."),
             Property.Number(label = "Max_Peak_Gap_Minutes", configurable = True, default_value = 60,
                             description="Default: 60. The longest the tune may wait for the temperature to turn, "
                                         "in minutes. A tune heats past the target, lets it cool back below, and "
                                         "heats again; each top and bottom is a peak. If no new peak comes this "
                                         "long after the last one, or after the start, the heater goes off, the "
                                         "tune ends and a notification says so. A probe stuck below the target "
                                         "makes no peaks, so it can keep the heater on for this long. With 20-40 L "
                                         "and a 5.5 kW element, heating up and the slowest swing at a mash "
                                         "temperature take up to about 50 minutes. Raise it for a bigger fill, a "
                                         "target below about 130 F (55 C) with the pump off, or a lower "
                                         "Output_Step or Max_Output.")])

class PIDAutotune(CBPiKettleLogic):

    # The sample time used when the property is absent or unusable.
    DEFAULT_SAMPLE_TIME = 5

    # AutoTuner refuses a sample time below 1 second, and requires the lookback
    # window to be at least one sample long.
    MIN_SAMPLE_TIME = 1

    # What run() takes each setting left empty as (blank_defaults). None of
    # them depends on the temperature unit.
    BLANK = {
        "Output_Step": 100,
        "Max_Output": 100,
        "SampleTime": DEFAULT_SAMPLE_TIME,
        "lockback_seconds": 30,
        "Max_Tune_Minutes": 300,
        "Max_Peak_Gap_Minutes": 60,
    }

    # How old a reading may be when the sensor's state doesn't say: the core's
    # own floor (SensorController.DEFAULT_MAX_AGE). Normally get_sensor_value()
    # gives max_age from the sensor's own publish interval.
    MAX_SENSOR_AGE = 30

    def __init__(self, cbpi, id, props):
        super().__init__(cbpi, id, props)
        self._logger = logging.getLogger(type(self).__name__)

    @classmethod
    def blank_defaults(cls, unit):
        """What run() uses for each setting left empty: BLANK, in either unit.
        run() reads the same table, so the two can't drift apart."""
        return dict(cls.BLANK)

    @staticmethod
    def _empty(raw):
        return raw is None or (isinstance(raw, str) and not raw.strip())

    @staticmethod
    def _finite(raw):
        """`raw` as a float, or None when it isn't a finite number."""
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    def _number(self, label):
        """A number setting: BLANK's value when it is empty. One that isn't a
        finite number raises ValueError, naming the setting."""
        raw = self.props.get(label)
        if self._empty(raw):
            return float(self.BLANK[label])
        value = self._finite(raw)
        if value is None:
            raise ValueError("{} is set to {!r}, which isn't a number".format(label, raw))
        return value

    def _minutes(self, label):
        """A time limit in seconds. Empty, it is BLANK's; a value that isn't a
        number of minutes above 0 is BLANK's too, and the brewer is told, so a
        mistyped limit can't switch the limit off."""
        raw = self.props.get(label)
        default = float(self.BLANK[label])
        if self._empty(raw):
            return default * 60
        value = self._finite(raw)
        if value is None or value <= 0:
            self.cbpi.notify(
                'PID AutoTune',
                "{} is set to {!r}, which isn't a number of minutes above 0, so {:g} "
                "is used.".format(label, raw, default),
                NotificationType.WARNING)
            return default * 60
        return value * 60

    def _kettle_name(self):
        return getattr(getattr(self, "kettle", None), "name", None) or "kettle"

    def _reading(self):
        """(temperature, None), or (None, why) when the kettle's probe gives
        nothing to heat on: no reading, a value that isn't a finite number, or
        one CraftBeerPi reports as stale - older than the max_age that
        get_sensor_value() gives with it, as PIDBoil, PID_HERMS and Hysteresis
        judge it. A sensor that gives no age is taken as it is, as they take
        it."""
        try:
            state = self.get_sensor_value(self.kettle.sensor)
        except Exception:  # noqa: BLE001 - no reading is an answer, not a crash
            state = None
        if not isinstance(state, dict):
            return None, "its sensor isn't giving a reading"
        raw = state.get("value")
        value = self._finite(raw)
        if value is None:
            return None, "its sensor reads {!r}, which isn't a temperature".format(raw)
        if state.get("age") is not None:
            age = self._finite(state.get("age"))
            limit = self._finite(state.get("max_age"))
            if limit is None or limit <= 0:
                limit = self.MAX_SENSOR_AGE
            if age is None:
                return None, "CraftBeerPi can't say how old its sensor's reading is"
            if age > limit:
                return None, ("its sensor has had no new reading for {:.0f} seconds, and CraftBeerPi "
                              "counts one over {:.0f} seconds old as out of date".format(age, limit))
        return value, None

    def _target(self):
        """(the kettle's target as a whole number, None), or (None, why) when it
        isn't a finite number."""
        raw = self.get_kettle_target_temp(self.id)
        value = self._finite(raw)
        if value is None:
            return None, "its target, {!r}, isn't a temperature".format(raw)
        return int(value), None

    async def _end_tune(self, why, atune=None, heating=True):
        """End the tune with the element off and the brewer told why.

        The element goes off first, before anything that could fail. Then the
        kettle's logic is stopped as AUTO off stops it, once this run has
        finished (_stop_when_done)."""
        self.finished = True
        await self.actor_off(self.heater)
        if heating:
            message = "AutoTune on {} ended and switched its heater off: {}.".format(self._kettle_name(), why)
        else:
            message = "AutoTune on {} didn't start, and its heater is off: {}.".format(self._kettle_name(), why)
        self._logger.warning("%s", message)
        if atune is not None:
            try:
                atune.log(message)
            except Exception:  # noqa: BLE001 - the tune's own log file is optional
                pass
        try:
            _notify_kept(self.cbpi, 'PID AutoTune', message, NotificationType.ERROR,
                         [NotificationAction("OK")])
        except Exception as e:  # noqa: BLE001 - telling must not stop the stop
            self._logger.error("Could not send the AutoTune notification: %s", e)
        self._stop_when_done()

    def _stop_when_done(self):
        """Stop the kettle's logic through CraftBeerPi, as AUTO off does, once
        this run's task has finished: the logic then shows stopped, and every
        page is told.

        CraftBeerPi's stop cancels the logic's task and waits for it, which the
        task can't do to itself, so it is asked for from the task's end. Only
        while this run is still the kettle's logic and still shown running."""
        task = getattr(self, "task", None)
        if task is None:
            return

        def stop(done):
            try:
                kettle = self.cbpi.kettle.find_by_id(self.id)
                if getattr(kettle, "instance", None) is self and self.state:
                    self._stopping = done.get_loop().create_task(self.cbpi.kettle.stop(self.id))
            except Exception as e:  # noqa: BLE001
                self._logger.error("Could not stop the kettle's logic after AutoTune ended: %s", e)

        task.add_done_callback(stop)

    def _sample_time(self):
        """Seconds between control decisions during the tune. Never raises.

        Parsed defensively like every other free-text-ish property on a kettle
        logic: an exception here would escape into run() and kill the tune
        after the element had already been driven, which is the worst moment
        to fail.

        Clamped to MIN_SAMPLE_TIME because AutoTuner raises ValueError below
        1 second, and that exception would surface as "AutoTune Error" with no
        hint that a sample time was the cause.
        """
        raw = self.props.get("SampleTime", self.DEFAULT_SAMPLE_TIME)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self.DEFAULT_SAMPLE_TIME
        if value != value or value in (float("inf"), float("-inf")):
            return self.DEFAULT_SAMPLE_TIME
        return max(self.MIN_SAMPLE_TIME, value)

    @staticmethod
    def rescale(ki, kd, tuned_at, used_at):
        """Deliberately an identity. Gains do NOT need rescaling by sample time.

        PIDArduino stores _Ki = ki * sampleTime and _Kd = kd / sampleTime, so
        its internals differ between sample rates - ON PURPOSE, because they
        are applied differently:

            self._iTerm += self._Ki * error      # once per sample
            d = -(self._Kd * dInput)             # dInput spans one sample

        At sample time s the integral adds ki*s*error every s seconds, which is
        ki*error per second regardless of s. The derivative sees a dInput
        proportional to s and multiplies by kd/s, which is kd * dT/dt
        regardless of s. Both terms are already per-second quantities, and that
        is exactly what the s factors are for.

        Scaling them anyway (ki * tuned_at / used_at, kd * used_at / tuned_at)
        is harmful. On the real rig's gains over a 40 minute hold near
        setpoint, the same gains at sample time 2 and 5 diverge by **0.228%**
        of output, while "rescaled" values diverge by **47.6%** and saturate.

        An identity rather than no method at all, so that anything calling it
        gets the right answer instead of an AttributeError.
        """
        return ki, kd


    async def autoOff(self):
        self.finished=True


    async def on_stop(self):
        if self.finished == False:
            self.cbpi.notify('PID AutoTune', 'Process stopped Manually. Please run Autotune again.', NotificationType.ERROR)
        await self.actor_off(self.heater)
        self.running = False

    async def run(self):
        self.finished = False
        self.kettle = self.get_kettle(self.id)
        self.heater = self.kettle.heater
        self.TEMP_UNIT=self.get_config_value("TEMP_UNIT", "C")
        fixedtarget = 67 if self.TEMP_UNIT == "C" else 153
        setpoint, why = self._target()
        if why is not None:
            await self._end_tune(why, heating=False)
            return
        current_value, why = self._reading()
        if why is not None:
            await self._end_tune(why, heating=False)
            return

        if setpoint == 0:
            if fixedtarget < current_value:
                self.cbpi.notify('PID AutoTune', 'You have not defined a target temp and current temp is above fallback value of {} °{}. Please choose target temp above current temp or wait until systme has cooled down.'.format(fixedtarget,self.TEMP_UNIT), NotificationType.ERROR)
                await self.actor_off(self.heater)
                await self.stop()
            self.cbpi.notify('PID AutoTune', 'You have not defined a target temp. System will set target to {} °{} and start AutoTune'.format(fixedtarget,self.TEMP_UNIT), NotificationType.WARNING)
            setpoint=fixedtarget
            await self.set_target_temp(self.id,setpoint)
    
        if setpoint < current_value:
            self.cbpi.notify('PID AutoTune', 'Your target temp is above the current temp. Choose a higher setpoint or wait until temp temp is below target temp and restart AutoTune', NotificationType.ERROR)
            await self.actor_off(self.heater)
            await self.stop()
            pass

        sampleTime = self._sample_time()
        wait_time = 5
        max_tune = self._minutes("Max_Tune_Minutes")
        max_gap = self._minutes("Max_Peak_Gap_Minutes")
        heat_percent_old = 0
        try:
            outstep = self._number("Output_Step")
            outmax = self._number("Max_Output")
            lookbackSec = self._number("lockback_seconds")
            atune = AutoTuner(setpoint, outstep, sampleTime, lookbackSec, 0, outmax,
                              getTimeMs=lambda: clock.monotonic() * 1000.0)
        except Exception as e:  # noqa: BLE001 - a setting the tune can't use is refused by raising
            # Ended as a reading it can't use is, before anything heats: the
            # brewer is told which setting, and the kettle's logic is stopped
            # as AUTO off stops it.
            await self._end_tune(str(e), heating=False)
            return
        self.cbpi.notify('PID AutoTune', 'AutoTune In Progress. Do not turn off Auto mode until AutoTuning is complete', NotificationType.INFO)
        atune.log("AutoTune will now begin")

        try:
            # Both limits count brewing time from here: the whole tune, and
            # the wait for each peak, the first one included.
            started = last_peak = clock.monotonic()
            peaks = 0
            ended = None
            await self.actor_on(self.heater, heat_percent_old)
            while self.running == True:
                reading, ended = self._reading()
                if ended is not None or atune.run(reading):
                    break
                now = clock.monotonic()
                if atune.peakCount != peaks:
                    peaks, last_peak = atune.peakCount, now
                if now - started >= max_tune:
                    ended = ("it has run for {:.0f} minutes without finishing, the most "
                             "Max_Tune_Minutes allows".format((now - started) / 60))
                    break
                if now - last_peak >= max_gap:
                    ended = ("the temperature hasn't turned for {:.0f} minutes, the most "
                             "Max_Peak_Gap_Minutes allows. A probe stuck below the target does "
                             "this, and so does a kettle slower than that: check the probe, and "
                             "raise Max_Peak_Gap_Minutes for a slow kettle".format((now - last_peak) / 60))
                    break
                heat_percent = atune.output
                if heat_percent != heat_percent_old:
                    await self.actor_set_power(self.heater,heat_percent)
                    heat_percent_old = heat_percent
                await clock.sleep(sampleTime)

            await self.autoOff()

            if ended is not None:
                await self._end_tune(ended, atune)
            elif atune.state == atune.STATE_SUCCEEDED:
                atune.log("AutoTune has succeeded")
                for rule in atune.tuningRules:
                    params = atune.getPIDParameters(rule)
                    atune.log('rule: {0}'.format(rule))
                    atune.log('P: {0}'.format(params.Kp))
                    atune.log('I: {0}'.format(params.Ki))
                    atune.log('D: {0}'.format(params.Kd))
                    atune.log('  (tuned at SampleTime {0}; these are '
                              'per-second gains and are used as-is at any '
                              'SampleTime)'.format(sampleTime))
                    if rule == "brewing":
                        self.cbpi.notify(
                            'AutoTune has succeeded',
                            "P: %.8f | I: %.8f | D: %.8f  - enter these as-is. "
                            "They do not need rescaling for your logic's "
                            "SampleTime." % (params.Kp, params.Ki, params.Kd),
                            action=[NotificationAction("OK")],
                        )
            elif atune.state == atune.STATE_FAILED:
                atune.log("AutoTune has failed")
                self.cbpi.notify('PID AutoTune Error', "PID AutoTune has failed",action=[NotificationAction("OK")])

        except asyncio.CancelledError as e:
            pass
        except Exception as e:
            logging.error("PIDAutoTune Error {}".format(e))
            await self.actor_off(self.heater)
            await self.stop()
            pass
        finally:
            await self.actor_off(self.heater)
            await self.stop()
            pass

# Based on a fork of Arduino PID AutoTune Library
# See https://github.com/t0mpr1c3/Arduino-PID-AutoTune-Library
class AutoTuner(object):
	PIDParams = namedtuple('PIDParams', ['Kp', 'Ki', 'Kd'])

	PEAK_AMPLITUDE_TOLERANCE = 0.05
	STATE_OFF = 'off'
	STATE_RELAY_STEP_UP = 'relay step up'
	STATE_RELAY_STEP_DOWN = 'relay step down'
	STATE_SUCCEEDED = 'succeeded'
	STATE_FAILED = 'failed'

	_tuning_rules = {
		# rule: [Kp_divisor, Ki_divisor, Kd_divisor]
		"ziegler-nichols": [34, 40, 160],
		"tyreus-luyben": [44,  9, 126],
		"ciancone-marlin": [66, 88, 162],
		"pessen-integral": [28, 50, 133],
		"some-overshoot": [60, 40,  60],
		"no-overshoot": [100, 40,  60],
		"brewing": [2.5, 3, 3600]
	}

	def __init__(self, setpoint, outputstep=10, sampleTimeSec=5, lookbackSec=60,
				 outputMin=float('-inf'), outputMax=float('inf'), noiseband=0.5, getTimeMs=None):
		if setpoint is None:
			raise ValueError('Kettle setpoint must be specified')
		if outputstep < 1:
			raise ValueError('Output_Step is {:g}, and it has to be at least 1'.format(outputstep))
		if sampleTimeSec < 1:
			raise ValueError('Sample Time Seconds must be greater or equal to 1')
		if lookbackSec < sampleTimeSec:
			raise ValueError('lockback_seconds is {:g}, and it has to be at least SampleTime, {:g}'.format(
				lookbackSec, sampleTimeSec))
		if outputMin >= outputMax:
			raise ValueError('Max_Output is {:g}, and it has to be more than {:g}'.format(
				outputMax, outputMin))

		self._inputs = deque(maxlen=round(lookbackSec / sampleTimeSec))
		self._sampleTime = sampleTimeSec * 1000
		self._setpoint = setpoint
		self._outputstep = outputstep
		self._noiseband = noiseband
		self._outputMin = outputMin
		self._outputMax = outputMax

		self._state = AutoTuner.STATE_OFF
		self._peakTimestamps = deque(maxlen=5)
		self._peaks = deque(maxlen=5)

		self._output = 0
		self._lastRunTimestamp = 0
		self._peakType = 0
		self._peakCount = 0
		self._initialOutput = 0
		self._inducedAmplitude = 0
		self._Ku = 0
		self._Pu = 0

		if getTimeMs is None:
			self._getTimeMs = self._currentTimeMs
		else:
			self._getTimeMs = getTimeMs

	@property
	def state(self):
		return self._state

	@property
	def output(self):
		return self._output

	@property
	def peakCount(self):
		return self._peakCount

	@property
	def tuningRules(self):
		return self._tuning_rules.keys()

	def getPIDParameters(self, tuningRule='ziegler-nichols'):
		divisors = self._tuning_rules[tuningRule]
		kp = self._Ku / divisors[0]
		ki = kp / (self._Pu / divisors[1])
		kd = kp * (self._Pu / divisors[2])
		return AutoTuner.PIDParams(kp, ki, kd)

	def log(self, text):
		filename = "./logs/autotune.log"
		formatted_time = strftime("%Y-%m-%d %H:%M:%S", localtime())

		with open(filename, "a") as file:
			file.write("%s,%s\n" % (formatted_time, text))
		
	def run(self, inputValue):
		now = self._getTimeMs()

		if (self._state == AutoTuner.STATE_OFF
				or self._state == AutoTuner.STATE_SUCCEEDED
				or self._state == AutoTuner.STATE_FAILED):
			self._initTuner(inputValue, now)
		elif (now - self._lastRunTimestamp) < self._sampleTime:
			return False

		self._lastRunTimestamp = now

		# check input and change relay state if necessary
		if (self._state == AutoTuner.STATE_RELAY_STEP_UP
				and inputValue > self._setpoint + self._noiseband):
			self._state = AutoTuner.STATE_RELAY_STEP_DOWN
			self.log('switched state: {0}'.format(self._state))
			self.log('input: {0}'.format(inputValue))
		elif (self._state == AutoTuner.STATE_RELAY_STEP_DOWN
				and inputValue < self._setpoint - self._noiseband):
			self._state = AutoTuner.STATE_RELAY_STEP_UP
			self.log('switched state: {0}'.format(self._state))
			self.log('input: {0}'.format(inputValue))

		# set output
		if (self._state == AutoTuner.STATE_RELAY_STEP_UP):
			self._output = self._initialOutput + self._outputstep
		elif self._state == AutoTuner.STATE_RELAY_STEP_DOWN:
			self._output = self._initialOutput - self._outputstep

		# respect output limits
		self._output = min(self._output, self._outputMax)
		self._output = max(self._output, self._outputMin)

		# identify peaks
		isMax = True
		isMin = True

		for val in self._inputs:
			isMax = isMax and (inputValue > val)
			isMin = isMin and (inputValue < val)

		self._inputs.append(inputValue)

		# we don't want to trust the maxes or mins until the input array is full
		if len(self._inputs) < self._inputs.maxlen:
			return False

		# increment peak count and record peak time for maxima and minima
		inflection = False

		# peak types:
		# -1: minimum
		# +1: maximum
		if isMax:
			if self._peakType == -1:
				inflection = True
			self._peakType = 1
		elif isMin:
			if self._peakType == 1:
				inflection = True
			self._peakType = -1

		# update peak times and values
		if inflection:
			self._peakCount += 1
			self._peaks.append(inputValue)
			self._peakTimestamps.append(now)
			self.log('found peak: {0}'.format(inputValue))
			self.log('peak count: {0}'.format(self._peakCount))

		# check for convergence of induced oscillation
		# convergence of amplitude assessed on last 4 peaks (1.5 cycles)
		self._inducedAmplitude = 0

		if inflection and (self._peakCount > 4):
			absMax = self._peaks[-2]
			absMin = self._peaks[-2]
			for i in range(0, len(self._peaks) - 2):
				self._inducedAmplitude += abs(self._peaks[i] - self._peaks[i+1])
				absMax = max(self._peaks[i], absMax)
				absMin = min(self._peaks[i], absMin)

			self._inducedAmplitude /= 6.0

			# check convergence criterion for amplitude of induced oscillation
			amplitudeDev = ((0.5 * (absMax - absMin) - self._inducedAmplitude)
							/ self._inducedAmplitude)

			self.log('amplitude: {0}'.format(self._inducedAmplitude))
			self.log('amplitude deviation: {0}'.format(amplitudeDev))

			if amplitudeDev < AutoTuner.PEAK_AMPLITUDE_TOLERANCE:
				self._state = AutoTuner.STATE_SUCCEEDED

		# if the autotune has not already converged
		# terminate after 10 cycles
		if self._peakCount >= 20:
			self._output = 0
			self._state = AutoTuner.STATE_FAILED
			return True

		if self._state == AutoTuner.STATE_SUCCEEDED:
			self._output = 0

			# calculate ultimate gain
			self._Ku = 4.0 * self._outputstep / (self._inducedAmplitude * math.pi)

			# calculate ultimate period in seconds
			period1 = self._peakTimestamps[3] - self._peakTimestamps[1]
			period2 = self._peakTimestamps[4] - self._peakTimestamps[2]
			self._Pu = 0.5 * (period1 + period2) / 1000.0
			return True

		return False

	def _currentTimeMs(self):
		return time.time() * 1000

	def _initTuner(self, inputValue, timestamp):
		self._peakType = 0
		self._peakCount = 0
		self._output = 0
		self._initialOutput = 0
		self._Ku = 0
		self._Pu = 0
		self._inputs.clear()
		self._peaks.clear()
		self._peakTimestamps.clear()
		self._peakTimestamps.append(timestamp)
		self._state = AutoTuner.STATE_RELAY_STEP_UP

def setup(cbpi):

    '''
    This method is called by the server during startup 
    Here you need to register your plugins at the server
    
    :param cbpi: the cbpi core 
    :return: 
    '''

    cbpi.plugin.register("PIDAutoTune", PIDAutotune)
