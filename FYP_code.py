#!/usr/bin/env python3
import time
import numpy as np
import cv2
import RPi.GPIO as GPIO
import pyzbar.pyzbar as pyzbar
import heapq
import matplotlib.pyplot as plt
from picamera2 import Picamera2
import os
import json
import logging
from datetime import datetime
import smbus
import collections
import statistics

# ============================ CONFIGURATION ============================
CONFIG_FILE = "robot_config.json"

DEFAULT_CONFIG = {
    "motor_settings": {
        "forward_speed": 70,
        "turn_speed": 60,  # Base turn speed for turning maneuvers
        "ramp_duration": 0.2,
        "turn_duration": 0.5,
        "invert_left_motor": False,
        "invert_right_motor": False,
        "balance_factor": 0.99,
        "balance_right": False
    },
    "camera_settings": {
        "resolution": [320, 240],
        "qr_timeout": 15
    },
    "navigation_settings": {
        "grid_spacing": 15,
        "path_simplification": True,
        "turn_threshold_degrees": 10,
        "safety_weight": 1.0,
        "min_grid_density": 0.5,
        "waypoint_delay": 2.0,
        "path_speed_factor": 0.7
    },
    "pid_settings": {
        "kp": 1.0,
        "ki": 0.1,
        "kd": 0.01,
        "max_integral": 50
    },
    "pid_turn_settings": {
        "kp": 0.8,
        "ki": 0.02,
        "kd": 0.15
    },
    "testing_mode": False,
    "pins": {
        "motor1_dir": 22,
        "motor1_pwm": 23,
        "motor2_dir": 24,
        "motor2_pwm": 25,
        "battery_monitor": 4,
        "ultrasonic_trigger": 16,
        "ultrasonic_echo": 26,
        "encoder_left_a": 5,
        "encoder_left_b": 6,
        "encoder_right_a": 17,
        "encoder_right_b": 18
    },
    "ultrasonic_settings": {
        "safe_distance_cm": 10,
        "check_interval": 0.5
    },
    "qr_locations": {
        "Entry": [60, 380],
        "Office1": [105, 635],
        "Office2": [280, 635],
        "Office3": [480, 635],
        "Office4": [660, 635],
        "Lounge1": [105, 125],
        "Lounge2": [285, 125],
        "Toilet1": [660, 280],
        "Toilet2": [660, 100]
    },
    "map_path": "/home/egallagher/FYP/Final_Map.png",
    "log_path": "robot_logs/",
    "encoder_ticks_per_meter": 15500,
    "encoder_ticks_per_degree": 11.6,
    "gyro_settings": {
        "i2c_address": 0x52,
        "gyro_scale": 20.0,
        "max_yaw_dps": 200,
        "still_threshold": 0.5,
        "reset_time": 3,
        "calibration_samples": 50,
        "samples_per_reading": 5,
        "turn_tolerance_degrees": 5,
        "adjustment_attempts": 3,
        "gyro_factor": 1.0    # Fudge factor for gyro integration (default 1.0)
    }
}

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)
                print(f"Configuration loaded from {CONFIG_FILE}")
                return config
        else:
            with open(CONFIG_FILE, 'w') as file:
                json.dump(DEFAULT_CONFIG, file, indent=4)
            print(f"Default configuration created at {CONFIG_FILE}")
            return DEFAULT_CONFIG
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return DEFAULT_CONFIG

CONFIG = load_config()

# ============================ MAP DIMENSIONS ============================
MAP_SIZE_PIXELS = 763
MAP_SIZE_METERS = 2.4
PIXELS_PER_METER = MAP_SIZE_PIXELS / MAP_SIZE_METERS

def pixels_to_meters(pixels):
    return pixels / PIXELS_PER_METER

def meters_to_pixels(meters):
    return meters * PIXELS_PER_METER

# ============================ LOGGING SETUP ============================
if not os.path.exists(CONFIG["log_path"]):
    os.makedirs(CONFIG["log_path"])
log_filename = os.path.join(CONFIG["log_path"], f"robot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RobotNavigator")
logger.info("Navigation system starting up")

# ============================ MOTOR CONTROL SETUP ============================
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

MOTOR1_DIR = CONFIG["pins"]["motor1_dir"]
MOTOR1_PWM = CONFIG["pins"]["motor1_pwm"]
MOTOR2_DIR = CONFIG["pins"]["motor2_dir"]
MOTOR2_PWM = CONFIG["pins"]["motor2_pwm"]
BATTERY_PIN = CONFIG["pins"]["battery_monitor"]
ULTRASONIC_TRIGGER = CONFIG["pins"]["ultrasonic_trigger"]
ULTRASONIC_ECHO = CONFIG["pins"]["ultrasonic_echo"]
ENCODER_LEFT_A = CONFIG["pins"]["encoder_left_a"]
ENCODER_LEFT_B = CONFIG["pins"]["encoder_left_b"]
ENCODER_RIGHT_A = CONFIG["pins"]["encoder_right_a"]
ENCODER_RIGHT_B = CONFIG["pins"]["encoder_right_b"]

GPIO.setup(MOTOR1_DIR, GPIO.OUT)
GPIO.setup(MOTOR1_PWM, GPIO.OUT)
GPIO.setup(MOTOR2_DIR, GPIO.OUT)
GPIO.setup(MOTOR2_PWM, GPIO.OUT)
GPIO.setup(BATTERY_PIN, GPIO.IN)
GPIO.setup(ULTRASONIC_TRIGGER, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)
GPIO.setup(ENCODER_LEFT_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENCODER_LEFT_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENCODER_RIGHT_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(ENCODER_RIGHT_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm1 = GPIO.PWM(MOTOR1_PWM, 25)
pwm2 = GPIO.PWM(MOTOR2_PWM, 25)
pwm1.start(0)
pwm2.start(0)

# Global state variables
robot_orientation = 0  # Updated by turns
motor_calibration = {"left_factor": 1.0, "right_factor": 1.0}
emergency_stop_flag = False

# ============================ ENCODER SETUP ============================
left_encoder_count = 0
right_encoder_count = 0

def left_encoder_callback(channel):
    global left_encoder_count
    left_encoder_count += 1

def right_encoder_callback(channel):
    global right_encoder_count
    right_encoder_count += 1

GPIO.add_event_detect(ENCODER_LEFT_A, GPIO.BOTH, callback=left_encoder_callback)
GPIO.add_event_detect(ENCODER_RIGHT_A, GPIO.BOTH, callback=right_encoder_callback)

# ============================ GYROSCOPE SETUP ============================
import smbus
import collections
import statistics
bus = smbus.SMBus(1)
gyro_address = CONFIG["gyro_settings"]["i2c_address"]
GYRO_SCALE = CONFIG["gyro_settings"]["gyro_scale"]
MAX_YAW_DPS = CONFIG["gyro_settings"]["max_yaw_dps"]
STILL_THRESHOLD = CONFIG["gyro_settings"]["still_threshold"]
RESET_TIME = CONFIG["gyro_settings"]["reset_time"]

yaw_buffer = collections.deque(maxlen=5)
yaw_bias = 0.0
still_count = 0
consecutive_errors = 0
gyro_initialized = False

# Global variable for integrated turn (for verification)
last_turn_gyro_change = None

def init_gyroscope():
    """Initialize the gyroscope (e.g. Wii MotionPlus)."""
    global gyro_initialized
    try:
        bus.write_byte_data(gyro_address, 0xFE, 0x05)
        time.sleep(0.1)
        logger.info("✅ Gyroscope initialized successfully")
        gyro_initialized = True
        return True
    except OSError as e:
        logger.error(f"❌ Gyroscope initialization error: {e}")
        gyro_initialized = False
        return False

def calibrate_gyro(samples=None, delay=0.05):
    """Calibrate the gyroscope bias while the robot is stationary."""
    global yaw_bias
    if samples is None:
        samples = CONFIG["gyro_settings"]["calibration_samples"]
    logger.info("Calibrating gyroscope... Keep the robot still.")
    readings = []
    for i in range(samples):
        try:
            data = bus.read_i2c_block_data(gyro_address, 0x00, 6)
            yaw_raw = ((data[3] >> 2) << 8) + data[0]
            yaw_dps = (yaw_raw - 8192) / GYRO_SCALE
            readings.append(yaw_dps)
            time.sleep(delay)
        except Exception as e:
            logger.warning(f"Gyro calibration sample {i} error: {e}")
            continue
    if readings:
        yaw_bias = statistics.median(readings)
        stdev = statistics.stdev(readings) if len(readings) > 1 else 0
        logger.info(f"Gyro calibrated: bias={yaw_bias:.2f}°/s, stdev={stdev:.2f}°/s")
    else:
        logger.warning("No valid gyro samples collected during calibration")
    return yaw_bias

def read_gyro():
    """Read the gyroscope, filter the value, and apply drift correction."""
    global consecutive_errors, still_count, yaw_bias
    try:
        data = bus.read_i2c_block_data(gyro_address, 0x00, 6)
        yaw_raw = ((data[3] >> 2) << 8) + data[0]
        yaw_dps = (yaw_raw - 8192) / GYRO_SCALE
        if abs(yaw_dps) > MAX_YAW_DPS:
            logger.debug(f"Gyro spike ignored: {yaw_dps:.2f}°/s")
            return None
        yaw_buffer.append(yaw_dps)
        yaw_filtered = statistics.median(yaw_buffer) if len(yaw_buffer) >= 3 else yaw_dps
        if abs(yaw_filtered - yaw_bias) < STILL_THRESHOLD:
            still_count += 1
        else:
            still_count = 0
        if still_count >= (RESET_TIME / 0.2):
            yaw_bias = yaw_filtered
            logger.debug("Gyro drift correction applied")
            still_count = 0
        consecutive_errors = 0
        return yaw_filtered - yaw_bias
    except OSError as e:
        consecutive_errors += 1
        if consecutive_errors % 5 == 0:
            logger.warning(f"Gyro read error: {e}")
        time.sleep(0.1)
        return None

if init_gyroscope():
    calibrate_gyro()
else:
    logger.error("Gyroscope initialization failed. Heading tracking unavailable.")

# ============================ MOTOR CONTROL FUNCTIONS ============================
def stop():
    logger.debug("Stopping motors")
    if not CONFIG["testing_mode"]:
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)
        GPIO.output(MOTOR1_DIR, GPIO.LOW)
        GPIO.output(MOTOR2_DIR, GPIO.LOW)
    time.sleep(0.1)

def apply_motor_speed(left_speed, right_speed, left_dir_high=True, right_dir_high=True):
    if CONFIG["testing_mode"]:
        logger.debug(f"(Test Mode) Motors would be set: Left {left_speed:.2f}% ({left_dir_high}), Right {right_speed:.2f}% ({right_dir_high})")
        return
    left_duty = left_speed * motor_calibration["left_factor"]
    right_duty = right_speed * motor_calibration["right_factor"]
    if left_dir_high == right_dir_high:
        balance_factor = CONFIG["motor_settings"].get("balance_factor", 1.0)
        if balance_factor < 1.0:
            if CONFIG["motor_settings"].get("balance_right", True):
                right_duty *= balance_factor
                logger.debug(f"Applying balance factor {balance_factor} to right motor")
            else:
                left_duty *= balance_factor
                logger.debug(f"Applying balance factor {balance_factor} to left motor")
    left_duty = max(0, min(100, left_duty))
    right_duty = max(0, min(100, right_duty))
    min_speed = 20
    if 0 < left_duty < min_speed:
        left_duty = min_speed
    if 0 < right_duty < min_speed:
        right_duty = min_speed
    actual_left_dir = left_dir_high
    actual_right_dir = right_dir_high
    if CONFIG["motor_settings"].get("invert_left_motor", False):
        actual_left_dir = not actual_left_dir
        logger.debug("Inverting left motor direction")
    if CONFIG["motor_settings"].get("invert_right_motor", False):
        actual_right_dir = not actual_right_dir
        logger.debug("Inverting right motor direction")
    GPIO.output(MOTOR1_DIR, GPIO.HIGH if actual_left_dir else GPIO.LOW)
    GPIO.output(MOTOR2_DIR, GPIO.HIGH if actual_right_dir else GPIO.LOW)
    logger.debug(f"Setting motors: Left {left_duty:.1f}% ({actual_left_dir}), Right {right_duty:.1f}% ({actual_right_dir})")
    pwm1.ChangeDutyCycle(left_duty)
    pwm2.ChangeDutyCycle(right_duty)

# ============================ PID CONTROLLER FOR STRAIGHT MOVEMENT ============================
previous_error = 0
integral = 0

def reset_pid():
    global previous_error, integral
    previous_error = 0
    integral = 0
    if hasattr(pid_controller, 'last_time'):
        delattr(pid_controller, 'last_time')
    if hasattr(pid_controller, 'last_left'):
        delattr(pid_controller, 'last_left')
    if hasattr(pid_controller, 'last_right'):
        delattr(pid_controller, 'last_right')
    if hasattr(pid_controller, 'last_derivative'):
        delattr(pid_controller, 'last_derivative')
    logger.debug("PID controller reset")

def pid_controller():
    global left_encoder_count, right_encoder_count, previous_error, integral
    current_time = time.time()
    if not hasattr(pid_controller, 'last_time'):
        pid_controller.last_time = current_time
        pid_controller.last_left = left_encoder_count
        pid_controller.last_right = right_encoder_count
        return CONFIG["motor_settings"]["forward_speed"], CONFIG["motor_settings"]["forward_speed"]
    dt = current_time - pid_controller.last_time
    if dt < 0.005:
        return CONFIG["motor_settings"]["forward_speed"], CONFIG["motor_settings"]["forward_speed"]
    left_rate = (left_encoder_count - pid_controller.last_left) / dt
    right_rate = (right_encoder_count - pid_controller.last_right) / dt
    position_error = left_encoder_count - right_encoder_count
    rate_error = left_rate - right_rate
    error = position_error * 0.3 + rate_error * 0.7
    deadband = 8
    if abs(error) < deadband:
        pid_controller.last_time = current_time
        pid_controller.last_left = left_encoder_count
        pid_controller.last_right = right_encoder_count
        return CONFIG["motor_settings"]["forward_speed"], CONFIG["motor_settings"]["forward_speed"]
    integral += error * dt
    integral = max(-CONFIG["pid_settings"]["max_integral"], min(CONFIG["pid_settings"]["max_integral"], integral))
    derivative = (error - previous_error) / dt if dt > 0 else 0
    if not hasattr(pid_controller, 'last_derivative'):
        pid_controller.last_derivative = derivative
    filtered_derivative = 0.7 * derivative + 0.3 * pid_controller.last_derivative
    pid_controller.last_derivative = filtered_derivative
    Kp = CONFIG["pid_settings"]["kp"] * 0.8
    Ki = CONFIG["pid_settings"]["ki"] * 0.5
    Kd = CONFIG["pid_settings"]["kd"] * 2.0
    output = (Kp * error) + (Ki * integral) + (Kd * filtered_derivative)
    max_correction = 25
    output = max(-max_correction, min(max_correction, output))
    previous_error = error
    pid_controller.last_time = current_time
    pid_controller.last_left = left_encoder_count
    pid_controller.last_right = right_encoder_count
    base_speed = CONFIG["motor_settings"]["forward_speed"]
    if output > 0:
        left_speed = base_speed - output
        right_speed = base_speed
    else:
        left_speed = base_speed
        right_speed = base_speed + output
    left_speed = max(base_speed * 0.4, min(base_speed * 1.2, left_speed))
    right_speed = max(base_speed * 0.4, min(base_speed * 1.2, right_speed))
    return left_speed, right_speed

def simple_wheel_sync_controller():
    global left_encoder_count, right_encoder_count
    if not hasattr(simple_wheel_sync_controller, 'last_time'):
        simple_wheel_sync_controller.last_time = time.time()
        simple_wheel_sync_controller.last_left = left_encoder_count
        simple_wheel_sync_controller.last_right = right_encoder_count
        return CONFIG["motor_settings"]["forward_speed"], CONFIG["motor_settings"]["forward_speed"]
    current_time = time.time()
    dt = current_time - simple_wheel_sync_controller.last_time
    if dt < 0.05:
        return CONFIG["motor_settings"]["forward_speed"], CONFIG["motor_settings"]["forward_speed"]
    left_distance = left_encoder_count - simple_wheel_sync_controller.last_left
    right_distance = right_encoder_count - simple_wheel_sync_controller.last_right
    left_speed = left_distance / dt
    right_speed = right_distance / dt
    simple_wheel_sync_controller.last_time = current_time
    simple_wheel_sync_controller.last_left = left_encoder_count
    simple_wheel_sync_controller.last_right = right_encoder_count
    base_speed = CONFIG["motor_settings"]["forward_speed"]
    if abs(left_speed - right_speed) < (max(left_speed, right_speed) * 0.05):
        return base_speed, base_speed
    logger.debug(f"Wheel sync: L speed={left_speed:.1f}, R speed={right_speed:.1f} ticks/s")
    left_output = base_speed
    right_output = base_speed
    if left_speed > 1 and right_speed > 1:
        if left_speed > right_speed:
            speed_ratio = right_speed / left_speed
            left_output = base_speed * (0.8 * speed_ratio + 0.2)
            logger.debug(f"Left wheel faster: ratio={speed_ratio:.2f}, adjusting to {left_output:.1f}%")
        else:
            speed_ratio = left_speed / right_speed
            right_output = base_speed * (0.8 * speed_ratio + 0.2)
            logger.debug(f"Right wheel faster: ratio={speed_ratio:.2f}, adjusting to {right_output:.1f}%")
    min_speed = base_speed * 0.6
    left_output = max(min_speed, left_output)
    right_output = max(min_speed, right_output)
    logger.debug(f"Wheel sync output: L={left_output:.1f}%, R={right_output:.1f}%")
    return left_output, right_output

# ============================ MOVEMENT FUNCTIONS ============================
def move_forward(duration=1.0):
    global left_encoder_count, right_encoder_count
    if emergency_stop_flag:
        return
    logger.info(f"Moving forward for {duration}s")
    reset_pid()
    left_encoder_count = 0
    right_encoder_count = 0
    start_time = time.time()
    while time.time() - start_time < duration:
        if emergency_stop_flag:
            return
        left_speed, right_speed = pid_controller()
        apply_motor_speed(left_speed, right_speed, True, True)
        time.sleep(0.01)
    stop()

def move_forward_distance(distance_meters):
    global left_encoder_count, right_encoder_count
    if emergency_stop_flag:
        return
    logger.info(f"Moving forward for {distance_meters:.2f} meters")
    ticks_per_meter = CONFIG["encoder_ticks_per_meter"]
    target_ticks = distance_meters * ticks_per_meter
    early_slowdown_point = target_ticks * 0.5
    mid_slowdown_point = target_ticks * 0.7
    final_approach_point = target_ticks * 0.9
    logger.debug(f"Slowdown points: early={early_slowdown_point:.0f}, mid={mid_slowdown_point:.0f}, final={final_approach_point:.0f} ticks")
    reset_pid()
    left_encoder_count = 0
    right_encoder_count = 0
    base_speed = CONFIG["motor_settings"]["forward_speed"]
    logger.debug("Applying initial boost to overcome inertia")
    apply_motor_speed(base_speed + 10, base_speed + 10, True, True)
    time.sleep(0.1)
    start_time = time.time()
    last_encoder_check = time.time()
    last_left_count = 0
    last_right_count = 0
    max_duration = distance_meters * 5
    while (left_encoder_count + right_encoder_count) / 2 < target_ticks:
        if emergency_stop_flag or (time.time() - start_time > max_duration):
            if time.time() - start_time > max_duration:
                logger.warning(f"Movement timed out after {time.time()-start_time:.1f} seconds")
            stop()
            return
        current_time = time.time()
        if current_time - last_encoder_check >= 0.5:
            encoder_delta_left = abs(left_encoder_count - last_left_count)
            encoder_delta_right = abs(right_encoder_count - last_right_count)
            if encoder_delta_left < 10 or encoder_delta_right < 10:
                logger.warning(f"Wheels may be stalled - L:{encoder_delta_left}, R:{encoder_delta_right} ticks in 0.5s")
                boost_speed = min(100, base_speed + 15)
                logger.debug(f"Applying boost: {boost_speed}%")
                apply_motor_speed(boost_speed, boost_speed, True, True)
                time.sleep(0.2)
            last_left_count = left_encoder_count
            last_right_count = right_encoder_count
            last_encoder_check = current_time
        left_speed, right_speed = pid_controller()
        average_ticks = (left_encoder_count + right_encoder_count) / 2
        if average_ticks > final_approach_point:
            slow_down_factor = 0.3
            logger.debug("Final approach slowdown")
        elif average_ticks > mid_slowdown_point:
            progress = (average_ticks - mid_slowdown_point) / (final_approach_point - mid_slowdown_point)
            slow_down_factor = 0.6 - (progress * 0.3)
            logger.debug(f"Medium slowdown: factor={slow_down_factor:.2f}")
        elif average_ticks > early_slowdown_point:
            progress = (average_ticks - early_slowdown_point) / (mid_slowdown_point - early_slowdown_point)
            slow_down_factor = 1.0 - (progress * 0.4)
            logger.debug(f"Early slowdown: factor={slow_down_factor:.2f}")
        else:
            slow_down_factor = 1.0
        left_final = left_speed * slow_down_factor
        right_final = right_speed * slow_down_factor
        apply_motor_speed(left_final, right_final, True, True)
        time.sleep(0.01)
    logger.debug("Applying braking pulse")
    apply_motor_speed(20, 20, False, False)
    time.sleep(0.05)
    stop()
    final_distance = ((left_encoder_count + right_encoder_count) / 2) / ticks_per_meter
    logger.info(f"Move complete. Traveled {final_distance:.2f}m of {distance_meters:.2f}m target")

def move_forward_distance_robust(distance_meters):
    global left_encoder_count, right_encoder_count
    if emergency_stop_flag:
        return
    logger.info(f"Moving forward for {distance_meters:.2f} meters with robust control")
    ticks_per_meter = CONFIG["encoder_ticks_per_meter"]
    target_ticks = distance_meters * ticks_per_meter
    early_slowdown_point = target_ticks * 0.6
    final_approach_point = target_ticks * 0.85
    left_encoder_count = 0
    right_encoder_count = 0
    base_speed = CONFIG["motor_settings"]["forward_speed"]
    logger.debug("Applying initial boost to overcome inertia")
    apply_motor_speed(base_speed + 15, base_speed + 15, True, True)
    time.sleep(0.15)
    start_time = time.time()
    last_encoder_check = time.time()
    last_left_count = 0
    last_right_count = 0
    max_duration = distance_meters * 5
    if hasattr(simple_wheel_sync_controller, 'last_time'):
        delattr(simple_wheel_sync_controller, 'last_time')
    while (left_encoder_count + right_encoder_count) / 2 < target_ticks:
        if emergency_stop_flag or (time.time() - start_time > max_duration):
            if time.time() - start_time > max_duration:
                logger.warning(f"Movement timed out after {time.time()-start_time:.1f} seconds")
            stop()
            return
        current_time = time.time()
        if current_time - last_encoder_check >= 0.5:
            encoder_delta_left = abs(left_encoder_count - last_left_count)
            encoder_delta_right = abs(right_encoder_count - last_right_count)
            if encoder_delta_left < 10 or encoder_delta_right < 10:
                logger.warning(f"Wheels may be stalled - L:{encoder_delta_left}, R:{encoder_delta_right} ticks in 0.5s")
                boost_speed = min(100, base_speed + 20)
                logger.debug(f"Applying emergency boost: {boost_speed}%")
                apply_motor_speed(boost_speed, boost_speed, True, True)
                time.sleep(0.25)
            last_left_count = left_encoder_count
            last_right_count = right_encoder_count
            last_encoder_check = current_time
        left_speed, right_speed = simple_wheel_sync_controller()
        average_ticks = (left_encoder_count + right_encoder_count) / 2
        if average_ticks > final_approach_point:
            slow_down_factor = 0.4
            logger.debug("Final approach slowdown")
        elif average_ticks > early_slowdown_point:
            progress = (average_ticks - early_slowdown_point) / (final_approach_point - early_slowdown_point)
            slow_down_factor = 0.9 - (progress * 0.5)
            logger.debug(f"Gradual slowdown: factor={slow_down_factor:.2f}")
        else:
            slow_down_factor = 1.0
        left_final = left_speed * slow_down_factor
        right_final = right_speed * slow_down_factor
        apply_motor_speed(left_final, right_final, True, True)
        time.sleep(0.01)
    logger.debug("Applying braking pulse")
    apply_motor_speed(25, 25, False, False)
    time.sleep(0.08)
    stop()
    final_distance = ((left_encoder_count + right_encoder_count) / 2) / ticks_per_meter
    logger.info(f"Move complete. Traveled {final_distance:.2f}m of {distance_meters:.2f}m target")
    logger.info(
        f"Final encoder counts - Left: {left_encoder_count}, Right: {right_encoder_count}, "
        f"Difference: {abs(left_encoder_count-right_encoder_count)} ticks "
        f"({abs(left_encoder_count-right_encoder_count)/max(left_encoder_count, right_encoder_count)*100:.1f}%)"
    )

# ============================ TURNING FUNCTIONS USING GYROSCOPE ============================
def turn_to_angle(target_angle):
    """
    Basic encoder-based turn. Determines turn direction from current orientation
    and uses a PID controller to turn.
    """
    global robot_orientation, left_encoder_count, right_encoder_count
    current_angle = robot_orientation % 360
    error_angle = ((target_angle - current_angle + 180) % 360) - 180
    if abs(error_angle) < 5:
        logger.info(f"Turn skipped: angle difference {abs(error_angle)}° too small")
        return
    turn_direction = "right" if error_angle > 0 else "left"
    angle_error = abs(error_angle)
    target_ticks = angle_error * CONFIG["encoder_ticks_per_degree"]
    left_encoder_count = 0
    right_encoder_count = 0
    pid_turn = CONFIG["pid_turn_settings"]
    Kp_turn = pid_turn["kp"]
    Ki_turn = pid_turn["ki"]
    Kd_turn = pid_turn["kd"]
    integral = 0
    previous_error = target_ticks
    last_time = time.time()
    base_turn_speed = CONFIG["motor_settings"]["turn_speed"]
    logger.info(f"Encoder turn: {current_angle}° → {target_angle}° ({angle_error}°), {turn_direction}, target_ticks={target_ticks:.2f}")
    while True:
        current_time = time.time()
        dt = current_time - last_time
        if dt <= 0:
            dt = 0.01
        if turn_direction == "right":
            progress = left_encoder_count
        else:
            progress = right_encoder_count
        error = target_ticks - progress
        if error <= 0:
            break
        integral += error * dt
        derivative = (error - previous_error) / dt
        output = Kp_turn * error + Ki_turn * integral + Kd_turn * derivative
        if error > 200:
            desired_speed = base_turn_speed
        elif error > 50:
            desired_speed = base_turn_speed * 0.7
        else:
            desired_speed = base_turn_speed * 0.4
        raw_pid_speed = max(20, min(base_turn_speed, output))
        turn_speed = min(desired_speed, raw_pid_speed)
        if turn_direction == "left":
            apply_motor_speed(turn_speed, turn_speed, left_dir_high=False, right_dir_high=True)
        else:
            apply_motor_speed(turn_speed, turn_speed, left_dir_high=True, right_dir_high=False)
        previous_error = error
        last_time = current_time
        time.sleep(0.01)
    stop()
    robot_orientation = target_angle % 360
    logger.info(f"Encoder turn complete. New orientation: {robot_orientation:.1f}°")

def turn_to_angle_gyro(target_angle, max_attempts=3):
    """
    Multi-attempt turn: Uses encoder-based turn followed by gyro verification.
    Modified to limit corrections to 5 degrees max and use 10% duty cycle for corrections.
    """
    global robot_orientation

    current_angle = robot_orientation % 360
    desired_turn = ((target_angle - current_angle + 180) % 360) - 180  # e.g. +60 means right turn, -60 means left turn
    angle_to_turn = abs(desired_turn)
    if angle_to_turn < 5:
        logger.info(f"Turn skipped: angle difference {angle_to_turn:.1f}° too small")
        return True

    # net_gyro_angle will accumulate the corrected gyro angle change (always with the proper sign)
    net_gyro_angle = 0.0
    tolerance = CONFIG["gyro_settings"]["turn_tolerance_degrees"]
    logger.info(f"Starting gyro-verified turn from {current_angle:.1f}° to {target_angle:.1f}° (desired turn: {desired_turn:.1f}°)")

    for attempt in range(1, max_attempts + 1):
        leftover = desired_turn - net_gyro_angle
        leftover_angle = abs(leftover)

        if leftover_angle < tolerance:
            logger.info(f"Turn verified within tolerance after {attempt-1} attempt(s).")
            robot_orientation = target_angle % 360
            return True

        # Limit correction angle to 5 degrees for correction attempts (not the first attempt)
        if attempt > 1 and leftover_angle > 5:
            leftover_angle = 5
            leftover = 5 if leftover > 0 else -5
            logger.info(f"Limiting correction to 5 degrees max")

        if leftover > 0:
            turn_direction = "right"
        else:
            turn_direction = "left"

        logger.info(f"--- Attempt {attempt}/{max_attempts}: leftover={leftover:.1f}° ({leftover_angle:.1f}° to go), direction={turn_direction}")

        ticks_to_turn = leftover_angle * CONFIG["encoder_ticks_per_degree"]
        global left_encoder_count, right_encoder_count
        left_encoder_count = 0
        right_encoder_count = 0

        # Integrate gyro over this correction attempt
        gyro_angle_change = 0.0
        last_gyro_time = time.time()

        pid_turn = CONFIG["pid_turn_settings"]
        Kp_turn = pid_turn["kp"]
        Ki_turn = pid_turn["ki"]
        Kd_turn = pid_turn["kd"]

        integral = 0
        previous_error = ticks_to_turn
        last_time = time.time()

        # Set lower turn speed for corrections (10% duty cycle)
        base_turn_speed = CONFIG["motor_settings"]["turn_speed"]
        correction_speed = 10 if attempt > 1 else base_turn_speed

        while True:
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.01

            # Integrate gyro readings at 100Hz
            gyro_dt = current_time - last_gyro_time
            if gyro_dt >= 0.01:
                gyro_rate = read_gyro()
                if gyro_rate is not None:
                    # We use absolute gyro reading so that we accumulate the magnitude.
                    gyro_angle_change += abs(gyro_rate) * gyro_dt * CONFIG["gyro_settings"].get("gyro_factor", 1.0)
                last_gyro_time = current_time

            if turn_direction == "right":
                progress = left_encoder_count
            else:
                progress = right_encoder_count

            error = ticks_to_turn - progress
            if error <= 0:
                break

            # For correction attempts, use a fixed lower speed
            if attempt > 1:
                desired_speed = correction_speed
            else:
                # Original granular speed control for first attempt
                progress_pct = progress / ticks_to_turn
                if progress_pct < 0.3:
                    desired_speed = base_turn_speed * 0.9
                elif progress_pct < 0.6:
                    desired_speed = base_turn_speed * 0.7
                elif progress_pct < 0.8:
                    desired_speed = base_turn_speed * 0.5
                elif progress_pct < 0.9:
                    desired_speed = base_turn_speed * 0.4
                else:
                    desired_speed = base_turn_speed * 0.3

                if error < 10:
                    desired_speed = 22
                elif error < 20:
                    desired_speed = 25

            integral += error * dt
            derivative = (error - previous_error) / dt
            output = Kp_turn * error + Ki_turn * integral + Kd_turn * derivative
            raw_pid_speed = max(20, min(base_turn_speed, output))

            # For correction attempts, make sure we don't exceed the correction speed
            if attempt > 1:
                turn_speed = min(correction_speed, raw_pid_speed)
            else:
                turn_speed = min(desired_speed, raw_pid_speed)

            if turn_direction == "left":
                apply_motor_speed(turn_speed, turn_speed, left_dir_high=False, right_dir_high=True)
            else:
                apply_motor_speed(turn_speed, turn_speed, left_dir_high=True, right_dir_high=False)

            previous_error = error
            last_time = current_time
            time.sleep(0.01)

        stop()
        time.sleep(0.3)
        logger.info(f"Gyro measured change this attempt: {gyro_angle_change:.1f}°")

        # Update net_gyro_angle using the absolute integrated value,
        # but apply the sign based on desired turn.
        if desired_turn > 0:  # Intended right turn
            net_gyro_angle += gyro_angle_change
        else:  # Intended left turn
            net_gyro_angle -= gyro_angle_change

        logger.info(f"After attempt {attempt}: net_gyro_angle={net_gyro_angle:.1f}°, desired turn={desired_turn:.1f}°")

        if abs(desired_turn - net_gyro_angle) < tolerance:
            logger.info(f"Turn verified within tolerance after {attempt} attempt(s).")
            robot_orientation = target_angle % 360
            return True

        logger.warning(f"Still outside tolerance after attempt {attempt}. Retrying correction.")

    logger.warning(f"Exceeded max attempts ({max_attempts}). Turn not verified.")
    robot_orientation = target_angle % 360
    return False

def turn_left():
    global robot_orientation
    if emergency_stop_flag:
        return
    logger.info(f"Turning left (90°) at {CONFIG['motor_settings']['turn_speed']}% speed")
    turn_to_angle_gyro((robot_orientation + 90) % 360)

def turn_right():
    global robot_orientation
    if emergency_stop_flag:
        return
    logger.info(f"Turning right (90°) at {CONFIG['motor_settings']['turn_speed']}% speed")
    turn_to_angle_gyro((robot_orientation - 90) % 360)

# ============================ QR CODE DETECTION ============================
from picamera2 import Picamera2
picam2 = Picamera2()
resolution = CONFIG["camera_settings"]["resolution"]
cam_config = picam2.create_preview_configuration(main={"size": (resolution[0], resolution[1]), "format": "RGB888"})
picam2.configure(cam_config)
picam2.start()
picam2.set_controls({"AwbEnable": True})

def detect_qr_code(timeout=None):
    if timeout is None:
        timeout = CONFIG["camera_settings"]["qr_timeout"]
    start_time = time.time()
    logger.info(f"Starting QR code detection with {timeout}s timeout")
    while time.time() - start_time < timeout:
        if emergency_stop_flag:
            return None, None
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded_objects = pyzbar.decode(gray)
        for obj in decoded_objects:
            qr_data = obj.data.decode("utf-8").strip()
            logger.info(f"QR Code Read: '{qr_data}'")
            location = qr_data
            orientation = None
            if ":" in qr_data:
                parts = qr_data.split(":")
                location = parts[0]
                try:
                    orientation = int(parts[1])
                    logger.info(f"QR Code contains orientation: {orientation}°")
                except:
                    logger.warning(f"Invalid orientation value in QR code: {parts[1]}")
            points = obj.polygon
            if points and len(points) > 3:
                pts = np.array([(p.x, p.y) for p in points], np.int32).reshape((-1, 1, 2))
                vis_frame = frame.copy()
                cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 3)
                text_x = min(p.x for p in points)
                text_y = max(p.y for p in points) + 30
                text_size, _ = cv2.getTextSize(qr_data, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis_frame, (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(vis_frame, qr_data, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(vis_frame, f"Time: {timestamp_str}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Location: {location}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if orientation is not None:
                    cv2.putText(vis_frame, f"Orientation: {orientation}°", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                qr_image_path = os.path.join(CONFIG["log_path"], f"qr_detected_{timestamp}.jpg")
                cv2.imwrite(qr_image_path, vis_frame)
                logger.debug(f"QR code image saved to {qr_image_path}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                qr_image_path = os.path.join(CONFIG["log_path"], f"qr_detected_{timestamp}.jpg")
                cv2.imwrite(qr_image_path, frame)
                logger.debug(f"QR code image saved to {qr_image_path} (no polygon available)")
            return location, orientation
        time.sleep(0.1)
        elapsed = time.time() - start_time
        if int(elapsed) % 2 == 0 and int(elapsed) != int(elapsed - 0.1):
            logger.debug(f"Searching for QR code... ({int(timeout - elapsed)}s left)")
    logger.warning("QR code detection timed out")
    return None, None

# ============================ MAP PROCESSING & PATHFINDING ============================
logger.info(f"Loading map from {CONFIG['map_path']}")
try:
    map_image = cv2.imread(CONFIG["map_path"], cv2.IMREAD_GRAYSCALE)
    if map_image is None:
        logger.error(f"Failed to load map from {CONFIG['map_path']}")
        map_image = np.ones((1000, 1000), dtype=np.uint8) * 255
    _, binary_map = cv2.threshold(map_image, 127, 255, cv2.THRESH_BINARY_INV)
    binary_map_int = (binary_map == 255).astype(np.uint8)
    inverted_for_dilation = 1 - binary_map_int
    edges = cv2.Canny(binary_map, 50, 150)
    corner_map = cv2.cornerHarris(edges, 5, 3, 0.04)
    corner_map = cv2.dilate(corner_map, None) > 0.01 * corner_map.max()
    corner_map = corner_map.astype(np.uint8)
    corner_kernel = np.ones((60, 60), np.uint8)
    dilated_corners = cv2.dilate(corner_map, corner_kernel, iterations=1)
    wall_kernel = np.ones((65, 65), np.uint8)
    dilated_walls = cv2.dilate(inverted_for_dilation, wall_kernel, iterations=1)
    combined_obstacles = np.maximum(dilated_walls, dilated_corners)
    binary_map_dilated = 1 - combined_obstacles
    binary_map = binary_map_dilated
    dist_transform = cv2.distanceTransform(binary_map.astype(np.uint8), cv2.DIST_L2, 5)
    min_dist = np.min(dist_transform)
    max_dist = np.max(dist_transform)
    logger.info(f"Distance transform range: {min_dist} to {max_dist}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dilated_map_path = os.path.join(CONFIG["log_path"], f"dilated_map_{timestamp}.png")
    cv2.imwrite(dilated_map_path, binary_map * 255)
    dist_map_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    dist_map_path = os.path.join(CONFIG["log_path"], f"distance_map_{timestamp}.png")
    cv2.imwrite(dist_map_path, dist_map_normalized.astype(np.uint8))
    logger.info("Map and safety gradient processed successfully")
except Exception as e:
    logger.error(f"Error processing map: {e}", exc_info=True)
    binary_map = np.ones((1000, 1000), dtype=np.uint8)
    dist_transform = np.ones_like(binary_map, dtype=np.float32) * 255

qr_locations_pixels = {k: tuple(v) for k, v in CONFIG["qr_locations"].items()}
logger.info(f"Loaded {len(qr_locations_pixels)} QR code locations")
grid_spacing = CONFIG["navigation_settings"]["grid_spacing"]
grid_nodes = {}

def is_valid_move(x, y, binary_map):
    h, w = binary_map.shape
    return 0 <= x < w and 0 <= y < h and binary_map[y, x] == 1

h, w = binary_map.shape
logger.info("Creating navigation grid with adaptive density...")
normalized_dist = dist_transform / max_dist if max_dist > 0 else np.ones_like(dist_transform)
min_grid_density = CONFIG["navigation_settings"]["min_grid_density"]
for y in range(0, h, grid_spacing):
    for x in range(0, w, grid_spacing):
        if not is_valid_move(x, y, binary_map):
            continue
        grid_nodes[(x, y)] = []
logger.info(f"Created {len(grid_nodes)} grid nodes with standard grid density")
additional_count = 0
half_spacing = grid_spacing // 2
for y in range(half_spacing, h, grid_spacing):
    for x in range(half_spacing, w, grid_spacing):
        if is_valid_move(x, y, binary_map):
            safety_value = normalized_dist[y, x]
            if safety_value < 0.5:
                grid_nodes[(x, y)] = []
                additional_count += 1
logger.info(f"Added {additional_count} additional nodes in areas near obstacles")
directions = [
    (0, grid_spacing), (0, -grid_spacing), (grid_spacing, 0), (-grid_spacing, 0),
    (half_spacing, half_spacing), (half_spacing, -half_spacing),
    (-half_spacing, half_spacing), (-half_spacing, -half_spacing)
]
for (x, y) in list(grid_nodes.keys()):
    for dx, dy in directions:
        neighbor = (x + dx, y + dy)
        if neighbor in grid_nodes:
            nx, ny = neighbor
            path_clear = True
            for t in np.linspace(0, 1, 10):
                px = int(x + t * (nx - x))
                py = int(y + t * (ny - y))
                if not is_valid_move(px, py, binary_map):
                    path_clear = False
                    break
            if path_clear:
                grid_nodes[(x, y)].append(neighbor)
for qr, (qx, qy) in qr_locations_pixels.items():
    if (qx, qy) not in grid_nodes:
        grid_nodes[(qx, qy)] = []
        logger.info(f"Added grid node for QR location: {qr} at ({qx}, {qy})")
    for node_x, node_y in list(grid_nodes.keys()):
        if (node_x, node_y) != (qx, qy):
            dist = np.sqrt((node_x - qx)**2 + (node_y - qy)**2)
            if dist <= grid_spacing * 1.5:
                path_clear = True
                for t in np.linspace(0, 1, 10):
                    px = int(qx + t * (node_x - qx))
                    py = int(qy + t * (node_y - qy))
                    if not is_valid_move(px, py, binary_map):
                        path_clear = False
                        break
                if path_clear:
                    grid_nodes[(qx, qy)].append((node_x, node_y))
                    grid_nodes[(node_x, node_y)].append((qx, qy))
logger.info(f"Final navigation grid has {len(grid_nodes)} nodes and all QR locations connected")

def a_star_with_grid(start, goal):
    logger.info(f"Finding path from {start} to {goal}")
    if start not in grid_nodes or goal not in grid_nodes:
        closest_start = min(grid_nodes.keys(), key=lambda n: np.linalg.norm(np.array(n) - np.array(start)))
        closest_goal = min(grid_nodes.keys(), key=lambda n: np.linalg.norm(np.array(n) - np.array(goal)))
        logger.warning(f"Start/goal not in grid. Using closest nodes: {closest_start} -> {closest_goal}")
        start, goal = closest_start, closest_goal
    queue = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    start_time = time.time()
    timeout = 10
    safety_weight = CONFIG["navigation_settings"]["safety_weight"]
    while queue and not emergency_stop_flag:
        if time.time() - start_time > timeout:
            logger.warning("Pathfinding timed out")
            return None
        current_cost, current = heapq.heappop(queue)
        if current == goal:
            break
        for neighbor in grid_nodes.get(current, []):
            dx = neighbor[0] - current[0]
            dy = neighbor[1] - current[1]
            move_cost = np.sqrt(dx*dx + dy*dy)
            neighbor_y, neighbor_x = int(neighbor[1]), int(neighbor[0])
            safety_value = normalized_dist[neighbor_y, neighbor_x]
            safety_cost = min(1.0 - safety_value, 0.8) * safety_weight
            new_cost = cost_so_far[current] + move_cost + safety_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h_cost = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                priority = new_cost + h_cost
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    if CONFIG["navigation_settings"]["path_simplification"] and len(path) > 2:
        path = simplify_path(path, binary_map)
        path = optimize_path(path, binary_map)
    logger.info(f"Path found with {len(path)} waypoints")
    return path

def simplify_path(path, binary_map):
    if len(path) <= 2:
        return path
    logger.info(f"Simplifying path from {len(path)} waypoints")
    def is_collision_free(p1, p2):
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        h, w = binary_map.shape
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while x0 != x1 or y0 != y1:
            if not (0 <= x0 < w and 0 <= y0 < h) or binary_map[y0, x0] == 0:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        max_visible = i + 1
        for j in range(len(path) - 1, i, -1):
            if is_collision_free(path[i], path[j]):
                max_visible = j
                break
        simplified.append(path[max_visible])
        i = max_visible
    logger.info(f"Path simplified to {len(simplified)} waypoints (with obstacle checking)")
    return simplified

def optimize_path(path, binary_map):
    if len(path) <= 3:
        return path
    logger.info(f"Optimizing path with {len(path)} waypoints")
    def is_collision_free(p1, p2):
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        h, w = binary_map.shape
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while x0 != x1 or y0 != y1:
            if not (0 <= x0 < w and 0 <= y0 < h) or binary_map[y0, x0] == 0:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True
    i = 0
    optimized_path = [path[0]]
    while i < len(path) - 1:
        current = path[i]
        next_point = path[i + 1]
        for j in range(len(path) - 1, i + 1, -1):
            if is_collision_free(current, path[j]):
                if i + 1 < j:
                    v1_x = next_point[0] - current[0]
                    v1_y = next_point[1] - current[1]
                    v2_x = path[j][0] - current[0]
                    v2_y = path[j][1] - current[1]
                    mag1 = np.sqrt(v1_x**2 + v1_y**2)
                    mag2 = np.sqrt(v2_x**2 + v2_y**2)
                    if mag1 > 0:
                        v1_x /= mag1; v1_y /= mag1
                    if mag2 > 0:
                        v2_x /= mag2; v2_y /= mag2
                    dot_product = v1_x * v2_x + v1_y * v2_y
                    if dot_product < 0.7 and mag2 > 1.5 * mag1:
                        continue
                next_point = path[j]
                i = j - 1
                break
        optimized_path.append(next_point)
        i += 1
    logger.info(f"Path optimized from {len(path)} to {len(optimized_path)} waypoints")
    return optimized_path

def visualize_path(path, save_path=None):
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CONFIG["log_path"], f"path_visualization_{timestamp}.png")
    vis_map = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    norm_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    heat_map = cv2.applyColorMap(norm_dist.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(vis_map, 0.7, heat_map, 0.3, 0)
    for node in grid_nodes.keys():
        if node not in qr_locations_pixels.values():
            cv2.circle(overlay, (int(node[0]), int(node[1])), 1, (0, 255, 255), -1)
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            start_point = (int(path[i][0]), int(path[i][1]))
            end_point = (int(path[i+1][0]), int(path[i+1][1]))
            cv2.line(overlay, start_point, end_point, (0, 0, 255), 3)
            cv2.circle(overlay, start_point, 5, (0, 255, 0), -1)
            cv2.putText(overlay, str(i), (start_point[0]+5, start_point[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(overlay, (int(path[-1][0]), int(path[-1][1])), 5, (0, 0, 255), -1)
    for name, (x, y) in qr_locations_pixels.items():
        cv2.putText(overlay, name, (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.circle(overlay, (x, y), 5, (255, 0, 255), -1)
    cv2.imwrite(save_path, overlay)
    logger.info(f"Path visualization saved to {save_path}")
    return save_path

def get_location_coordinates(location_name):
    if location_name in qr_locations_pixels:
        return qr_locations_pixels[location_name]
    for qr_key, coordinates in qr_locations_pixels.items():
        base_location = qr_key.split(':')[0] if ':' in qr_key else qr_key
        if base_location == location_name:
            return coordinates
    logger.warning(f"Location {location_name} not found in configuration. Using closest match.")
    closest_match = None
    for qr_key in qr_locations_pixels.keys():
        base_location = qr_key.split(':')[0] if ':' in qr_key else qr_key
        if closest_match is None or len(base_location) > len(closest_match):
            if location_name.startswith(base_location):
                closest_match = base_location
    if closest_match:
        return qr_locations_pixels[closest_match]
    return next(iter(qr_locations_pixels.values()))

def verify_arrival(destination):
    global robot_orientation
    logger.info(f"Verifying arrival at {destination}...")
    qr_data, orientation = detect_qr_code(timeout=5)
    if qr_data and qr_data.split(':')[0] == destination:
        logger.info(f"Arrival verified! Found QR code: {qr_data}")
        if orientation is not None:
            robot_orientation = orientation
            logger.info(f"Final orientation set to: {robot_orientation}°")
        return True
    if CONFIG["testing_mode"]:
        logger.info("Testing mode: Skipping QR verification rotation")
        return True
    logger.info("Initial scan failed. Starting comprehensive verification process...")
    for angle in [90, 180, 270, 0]:
        turn_to_angle_gyro(angle)
        time.sleep(0.5)
        qr_data, orientation = detect_qr_code(timeout=3)
        if qr_data and qr_data.split(':')[0] == destination:
            logger.info(f"Location verified after turning to {angle}°: {qr_data}")
            if orientation is not None:
                robot_orientation = orientation
            return True
    logger.info("90-degree scan failed. Trying 45-degree increments...")
    for angle in [45, 135, 225, 315]:
        turn_to_angle_gyro(angle)
        time.sleep(0.5)
        qr_data, orientation = detect_qr_code(timeout=2)
        if qr_data and qr_data.split(':')[0] == destination:
            logger.info(f"Location verified after fine turning to {angle}°: {qr_data}")
            if orientation is not None:
                robot_orientation = orientation
            return True
    logger.info("Angular scan failed. Trying small movements to find QR code...")
    move_forward_distance(0.1)
    qr_data, orientation = detect_qr_code(timeout=2)
    if qr_data and qr_data.split(':')[0] == destination:
        logger.info(f"Location verified after moving forward: {qr_data}")
        if orientation is not None:
            robot_orientation = orientation
        return True
    move_forward_distance(-0.1)
    move_forward_distance(-0.1)
    qr_data, orientation = detect_qr_code(timeout=2)
    if qr_data and qr_data.split(':')[0] == destination:
        logger.info(f"Location verified after moving backward: {qr_data}")
        if orientation is not None:
            robot_orientation = orientation
        return True
    move_forward_distance(0.1)
    logger.warning(f"Could not verify arrival at {destination} after exhaustive scanning")
    return False

# ============================ OBSTACLE AVOIDANCE ============================
def test_forward_ultrasonic():
    """
    Test function that drives the robot forward using the robust movement controller
    until an obstacle is detected within the safe distance by the ultrasonic sensor.
    """
    global emergency_stop_flag, left_encoder_count, right_encoder_count
    emergency_stop_flag = False

    safe_distance = CONFIG["ultrasonic_settings"]["safe_distance_cm"]
    logger.info(f"Starting ultrasonic obstacle test (stopping at {safe_distance}cm)")

    # Using components of the robust forward movement
    left_encoder_count = 0
    right_encoder_count = 0
    base_speed = CONFIG["motor_settings"]["forward_speed"]

    print("Robot will drive forward using robust movement and stop when an obstacle is detected within 10cm")
    print("Press Ctrl+C to abort the test")

    # Set a time limit for safety
    max_duration = 30  # seconds
    start_time = time.time()

    try:
        # Apply initial boost to overcome inertia (from move_forward_distance_robust)
        logger.debug("Applying initial boost to overcome inertia")
        apply_motor_speed(base_speed + 15, base_speed + 15, True, True)
        time.sleep(0.15)

        last_encoder_check = time.time()
        last_left_count = 0
        last_right_count = 0

        # Reset the wheel sync controller
        if hasattr(simple_wheel_sync_controller, 'last_time'):
            delattr(simple_wheel_sync_controller, 'last_time')

        # Main movement loop with obstacle checking
        while not emergency_stop_flag and (time.time() - start_time < max_duration):
            current_time = time.time()

            # Check for obstacles
            distance = get_distance()

            # Print distance every second
            if int(current_time) % 1 == 0 and int(current_time) != int(current_time - 0.1):
                print(f"Current distance: {distance:.1f}cm, Encoders - Left: {left_encoder_count}, Right: {right_encoder_count}")

            # Stop if obstacle detected
            if distance < safe_distance:
                logger.warning(f"Obstacle detected at {distance:.1f}cm! Stopping.")
                print(f"\nOBSTACLE DETECTED at {distance:.1f}cm")

                # Apply braking pulse (from move_forward_distance_robust)
                logger.debug("Applying braking pulse")
                apply_motor_speed(25, 25, False, False)
                time.sleep(0.08)
                stop()

                # Report distance traveled
                distance_meters = ((left_encoder_count + right_encoder_count) / 2) / CONFIG["encoder_ticks_per_meter"]
                logger.info(f"Travel stopped due to obstacle. Traveled {distance_meters:.2f}m")
                print(f"Travel distance: {distance_meters:.2f}m")

                return True

            # Check if wheels are stalled every 0.5 seconds
            if current_time - last_encoder_check >= 0.5:
                encoder_delta_left = abs(left_encoder_count - last_left_count)
                encoder_delta_right = abs(right_encoder_count - last_right_count)

                if encoder_delta_left < 10 or encoder_delta_right < 10:
                    logger.warning(f"Wheels may be stalled - L:{encoder_delta_left}, R:{encoder_delta_right} ticks in 0.5s")
                    boost_speed = min(100, base_speed + 20)
                    logger.debug(f"Applying emergency boost: {boost_speed}%")
                    apply_motor_speed(boost_speed, boost_speed, True, True)
                    time.sleep(0.25)

                last_left_count = left_encoder_count
                last_right_count = right_encoder_count
                last_encoder_check = current_time

            # Use simple wheel sync controller (from move_forward_distance_robust)
            left_speed, right_speed = simple_wheel_sync_controller()

            # Apply speeds
            apply_motor_speed(left_speed, right_speed, True, True)
            time.sleep(0.01)

        # If we reached the time limit
        if time.time() - start_time >= max_duration:
            logger.info("Test completed: time limit reached without detecting obstacles")
            print("\nTest completed: Maximum time reached without detecting obstacles")

            # Apply braking pulse
            logger.debug("Applying braking pulse")
            apply_motor_speed(25, 25, False, False)
            time.sleep(0.08)
            stop()

            # Report distance traveled
            distance_meters = ((left_encoder_count + right_encoder_count) / 2) / CONFIG["encoder_ticks_per_meter"]
            logger.info(f"Travel complete. Traveled {distance_meters:.2f}m")
            print(f"Travel distance: {distance_meters:.2f}m")

        return False

    except KeyboardInterrupt:
        logger.info("Test aborted by user")
        print("\nTest aborted by user")
        return False
    finally:
        # Make sure we stop the robot
        stop()
        print("Robot stopped")

def get_distance():
    GPIO.output(ULTRASONIC_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(ULTRASONIC_TRIGGER, False)
    pulse_start = time.time()
    pulse_end = time.time()
    while GPIO.input(ULTRASONIC_ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ULTRASONIC_ECHO) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance_cm = pulse_duration * 17150
    distance_cm = round(distance_cm, 2)
    return distance_cm

def obstacle_detected():
    distance = get_distance()
    logger.debug(f"Obstacle distance: {distance} cm")
    return distance < CONFIG["ultrasonic_settings"]["safe_distance_cm"]

def avoid_obstacle():
    logger.info("Obstacle detected! Avoiding...")
    stop()
    time.sleep(0.5)
    if np.random.rand() > 0.5:
        logger.info("Turning left to avoid obstacle")
        turn_left()
    else:
        logger.info("Turning right to avoid obstacle")
        turn_right()
    move_forward_distance(0.3)

def execute_path(path):
    """
    Modified execute_path function that prevents premature stopping when seeing destination QR code.
    QR verification is only performed at the final destination, not during travel.
    """
    global robot_orientation
    if emergency_stop_flag:
        return False

    waypoint_delay = CONFIG["navigation_settings"]["waypoint_delay"]
    path_speed_factor = CONFIG["navigation_settings"]["path_speed_factor"]

    original_forward_speed = CONFIG["motor_settings"]["forward_speed"]
    original_turn_speed = CONFIG["motor_settings"]["turn_speed"]
    CONFIG["motor_settings"]["forward_speed"] = int(original_forward_speed * path_speed_factor)
    CONFIG["motor_settings"]["turn_speed"] = int(original_turn_speed * path_speed_factor)

    logger.info(f"Starting navigation with {len(path)} waypoints")
    logger.info(f"Movement speed adjusted to {path_speed_factor*100:.0f}% (forward: {CONFIG['motor_settings']['forward_speed']}%, turn: {CONFIG['motor_settings']['turn_speed']}%)")
    logger.info(f"Waypoint delay set to {waypoint_delay} seconds")

    # Determine destination name before starting
    destination = None
    if len(path) > 0:
        for location_name, coords in qr_locations_pixels.items():
            if tuple(coords) == tuple(path[-1]):
                destination = location_name
                break

    if destination:
        logger.info(f"Destination identified as: {destination}")
        # Disable QR detection during navigation to prevent premature stopping
        logger.info(f"QR verification will ONLY occur at final destination ({destination})")

    try:
        for i in range(len(path) - 1):
            if emergency_stop_flag:
                return False

            current_x, current_y = path[i]
            next_x, next_y = path[i+1]

            waypoint_name = "START" if i == 0 else f"WAYPOINT {i}"
            current_location = None
            for loc, coords in qr_locations_pixels.items():
                if tuple(coords) == (current_x, current_y):
                    current_location = loc
                    waypoint_name = loc
                    logger.info(f"Current waypoint is at known location: {current_location}")
                    break

            print("\n" + "="*50)
            print(f"   {waypoint_name} ({current_x}, {current_y})")
            print(f"   → Next: {next_x}, {next_y} (Waypoint {i+1}/{len(path)-1})")
            print("="*50)

            if obstacle_detected() and not CONFIG["testing_mode"]:
                avoid_obstacle()
                current_position = path[i]
                goal_position = path[-1]
                logger.info("Recalculating path due to obstacle...")
                path = a_star_with_grid(current_position, goal_position)
                if not path:
                    logger.error("No valid path found after obstacle avoidance.")
                    return False
                visualize_path(path)
                continue

            logger.info(f"At waypoint {i}: ({current_x}, {current_y}) → ({next_x}, {next_y})")

            if i > 0 and waypoint_delay > 0:
                logger.info(f"Pausing at waypoint {i} for {waypoint_delay} seconds...")
                print(f"\n[PAUSED] At waypoint {i} ({waypoint_name}). Continue in {waypoint_delay} seconds...")
                time.sleep(waypoint_delay)

            dx = next_x - current_x
            dy = next_y - current_y
            angle = np.degrees(np.arctan2(-dy, dx)) % 360
            logger.info(f"Turning to angle: {angle:.1f}° to face next waypoint")
            turn_to_angle_gyro(angle)

            distance_pixels = np.sqrt(dx**2 + dy**2)
            distance_meters = pixels_to_meters(distance_pixels)
            logger.info(f"Moving forward for distance: {distance_meters:.2f} meters")

            # If this is the final destination and there's a QR code there,
            # use a modified approach to avoid stopping early
            if i == len(path) - 2 and destination:
                logger.info("Approaching final destination - disabling camera until arrival")
                # Save camera state
                camera_was_running = hasattr(picam2, 'camera_state') and picam2.camera_state
                if camera_was_running:
                    picam2.stop()
                    time.sleep(0.5)  # Give camera time to stop

                # Move to destination using encoders only
                move_forward_distance_robust(distance_meters)

                # Restore camera if needed
                if camera_was_running:
                    picam2.start()
                    time.sleep(0.5)  # Give camera time to start
            else:
                # Normal movement for intermediate waypoints
                move_forward_distance_robust(distance_meters)

            logger.info(f"Arrived at waypoint {i+1}/{len(path)-1}")

            if (i+1) % 3 == 0 and i < len(path)-2:
                continue_choice = input("\nContinue to next waypoint? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    logger.info("Navigation paused by user")
                    print("\nNavigation paused. Press Enter to continue or Ctrl+C to abort...")
                    input()

        print("\n" + "="*50)
        print(f"   DESTINATION REACHED: {destination}")
        print("="*50 + "\n")
        logger.info("Path execution complete!")

        if destination:
            logger.info("Now verifying arrival by finding destination QR code...")
            arrival_verified = verify_arrival(destination)
            if arrival_verified:
                logger.info("Navigation complete with verified arrival!")
            else:
                logger.warning("Navigation complete but arrival could not be verified")

        return True

    finally:
        CONFIG["motor_settings"]["forward_speed"] = original_forward_speed
        CONFIG["motor_settings"]["turn_speed"] = original_turn_speed
def toggle_testing_mode():
    CONFIG["testing_mode"] = not CONFIG["testing_mode"]
    print(f"Testing mode {'enabled' if CONFIG['testing_mode'] else 'disabled'}")
    logger.info(f"Testing mode {'enabled' if CONFIG['testing_mode'] else 'disabled'}")

def check_battery():
    try:
        battery_state = GPIO.input(BATTERY_PIN)
        logger.info(f"Battery state: {'OK' if battery_state else 'LOW'}")
        return battery_state
    except Exception as e:
        logger.warning(f"Battery check failed: {e}")
        return True

def navigate_to_destination():
    global emergency_stop_flag, robot_orientation
    emergency_stop_flag = False
    if not CONFIG["testing_mode"] and not check_battery():
        logger.warning("Battery low, navigation may be unreliable")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return False
    logger.info("Scanning for initial QR code...")
    max_attempts = 3
    start_location = None
    qr_orientation = None
    if CONFIG["testing_mode"]:
        print("\nTesting mode: Manual QR code entry")
        valid_locations = list(qr_locations_pixels.keys())
        print(f"Valid locations: {valid_locations}")
        start_location = input("Enter starting location: ").strip()
        if start_location not in qr_locations_pixels:
            logger.error(f"Invalid start location: {start_location}")
            return False
    else:
        for attempt in range(max_attempts):
            if emergency_stop_flag:
                return False
            start_location, qr_orientation = detect_qr_code()
            if start_location and start_location in qr_locations_pixels:
                break
            logger.warning(f"Attempt {attempt+1}/{max_attempts} failed. Retrying...")
        if not start_location or start_location not in qr_locations_pixels:
            logger.error("Could not determine starting position. Aborting.")
            return False
    logger.info(f"Start location identified: {start_location}")
    if qr_orientation is not None:
        robot_orientation = qr_orientation
        logger.info(f"Robot orientation set from QR code: {robot_orientation}°")
    else:
        logger.warning("No orientation data in QR code. Using current orientation.")
    valid_destinations = list(qr_locations_pixels.keys())
    print(f"Valid destinations: {valid_destinations}")
    goal_location = input("Enter Destination: ").strip()
    if goal_location not in qr_locations_pixels:
        logger.error(f"Invalid destination '{goal_location}'. Valid destinations are: {valid_destinations}")
        return False
    start_pos = qr_locations_pixels[start_location]
    goal_pos = qr_locations_pixels[goal_location]
    logger.info(f"Calculating path from {start_location} {start_pos} to {goal_location} {goal_pos}...")
    path = a_star_with_grid(start_pos, goal_pos)
    if not path or len(path) < 2:
        logger.error("Could not find a valid path. Please check the map.")
        return False
    logger.info(f"Computed path with {len(path)} waypoints")
    vis_path = visualize_path(path)
    print(f"Path visualization saved to: {vis_path}")
    if input("Ready to execute path? (y/n): ").lower() != 'y':
        logger.info("Navigation canceled by user")
        return False
    try:
        success = execute_path(path)
        if success:
            logger.info("Navigation Complete!")
            return True
        else:
            logger.warning("Navigation interrupted")
            return False
    except Exception as e:
        logger.error(f"Navigation error: {e}", exc_info=True)
        stop()
        return False

# ============================ DUMMY CALIBRATION STUBS ============================
def calibrate_motors():
    logger.info("Calibrate motors stub...")

def calibrate_encoders():
    logger.info("Calibrate encoders stub...")

def test_encoders_enhanced():
    logger.info("Test encoders (enhanced) stub...")

def troubleshoot_encoders():
    logger.info("Troubleshoot encoders stub...")

def find_motor_imbalance():
    logger.info("Find motor imbalance stub...")

def calibrate_turns():
    logger.info("Calibrate turns stub...")

def calibrate_straight_movement():
    logger.info("Calibrate straight movement stub...")

def emergency_stop():
    global emergency_stop_flag
    emergency_stop_flag = True
    stop()
    logger.warning("EMERGENCY STOP ACTIVATED")

# ============================ GYRO FACTOR CALIBRATION ROUTINE ============================
def get_stable_heading(duration=2.0, sample_rate=100):
    """
    Integrate the gyroscope angular velocity over a short time period to estimate heading.
    Returns the heading in degrees.
    """
    total_angle = 0.0
    sample_interval = 1.0 / sample_rate
    num_samples = int(duration * sample_rate)
    last_time = time.time()

    for _ in range(num_samples):
        gyro_rate = read_gyro()
        now = time.time()
        dt = now - last_time
        last_time = now
        if gyro_rate is not None:
            total_angle += gyro_rate * dt
        time.sleep(max(0, sample_interval - (time.time() - now)))

    return total_angle % 360

def calibrate_gyro_factor():
    """
    Run a calibration routine to adjust the gyro fudge factor (gyro_factor).
    The user is prompted to perform a turn at the normal navigation turn speed (30%).
    The routine then measures the gyro reading before and after the turn, and the user inputs
    the actual turn in degrees. The ratio of actual turn / gyro-measured turn becomes the gyro_factor,
    which is saved in the config.
    """
    global CONFIG
    print("=== Gyro Factor Calibration ===")
    print("Ensure the robot is on a flat surface.")
    # Set turn speed to navigation turn speed (30% as specified)
    normal_turn_speed = 30
    print(f"Setting turn speed to {normal_turn_speed}% for calibration.")
    input("Press Enter when ready to start the turn...")

    # Record pre-turn gyro heading using the existing get_stable_heading function
    pre_turn = get_stable_heading(num_samples=5, delay=0.1)
    print(f"Pre-turn gyro heading: {pre_turn:.1f}°")

    print("Perform the turn now (using your normal navigation method) and then press Enter when done...")
    input("Press Enter after completing the turn...")

    # Record post-turn gyro heading
    post_turn = get_stable_heading(num_samples=5, delay=0.1)
    print(f"Post-turn gyro heading: {post_turn:.1f}°")

    # Compute the gyro-measured turn using a minimal angle difference function
    def angle_diff(start, end):
        return ((end - start + 180) % 360) - 180

    gyro_measured = angle_diff(pre_turn, post_turn)
    print(f"Gyro measured change: {gyro_measured:.1f}°")

    # Ask the user for the actual turn performed (external measurement)
    actual_turn = float(input("Enter the actual turn in degrees (as measured externally): "))

    # Calculate the fudge factor (gyro_factor)
    if gyro_measured != 0:
        gyro_factor = actual_turn / abs(gyro_measured)
    else:
        gyro_factor = 1.0
    print(f"Calculated gyro_factor: {gyro_factor:.3f}")

    # Save the fudge factor to the config
    CONFIG["gyro_settings"]["gyro_factor"] = gyro_factor
    with open(CONFIG_FILE, 'w') as file:
        json.dump(CONFIG, file, indent=4)
    print("Gyro factor updated and saved in configuration.")
    return gyro_factor

# ============================ MAIN EXECUTION ============================
def main():
    print("\n==== Robot Navigation System ====")
    print("1. Start Navigation")
    print("2. Calibrate Motors")
    print("3. Calibrate Encoders")
    print("4. Test Movement")
    print("5. Scan QR Code")
    print("6. Test Ultrasonic Sensor")
    print("7. Test Encoders (Basic)")
    print("8. Test Encoders (Enhanced Debugging)")
    print("9. Troubleshoot Encoders (Advanced)")
    print("F. Forward Obstacle Detection Test") # New ultrasonic forward test
    print("R. Turn Test with Gyroscope Verification")
    print("G. Gyro Factor Calibration")
    print("X. Toggle Testing Mode (currently {})".format("ON" if CONFIG["testing_mode"] else "OFF"))
    print("0. Exit")
    choice = input("Select an option: ").strip()
    try:
        if choice == '1':
            navigate_to_destination()
        elif choice == '2':
            calibrate_motors()
        elif choice == '3':
            calibrate_encoders()
        elif choice == '4':
            print("Testing movements...")
            print("1. Move Forward 0.5m")
            print("2. Turn 90°")
            print("3. Full Rotation Test")
            movement_choice = input("Select movement test: ").strip()
            if movement_choice == '1':
                move_forward_distance(0.5)
            elif movement_choice == '2':
                current_angle = robot_orientation
                target_angle = (current_angle + 90) % 360
                turn_to_angle_gyro(target_angle)
            elif movement_choice == '3':
                for angle in [90, 180, 270, 0]:
                    turn_to_angle_gyro(angle)
                    time.sleep(0.5)
            print("Test complete")
        elif choice == '5':
            print("Scanning for QR code...")
            qr_data, orientation = detect_qr_code(timeout=10)
            if qr_data:
                print(f"Detected QR code: {qr_data}")
                if orientation is not None:
                    print(f"Orientation: {orientation}°")
                if qr_data in qr_locations_pixels:
                    print(f"Location: {qr_locations_pixels[qr_data]}")
                else:
                    print("Unknown location")
            else:
                print("No QR code detected")
        elif choice == '6':
            print("Testing ultrasonic sensor...")
            try:
                while True:
                    distance = get_distance()
                    print(f"Distance: {distance} cm")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Ultrasonic test stopped.")
        elif choice == '7':
            print("Testing encoders (Basic)...")
            print("Rotate the wheels to see encoder counts.")
            try:
                global left_encoder_count, right_encoder_count
                left_encoder_count = 0
                right_encoder_count = 0
                while True:
                    print(f"Left encoder: {left_encoder_count}, Right encoder: {right_encoder_count}")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Encoder test stopped.")
        elif choice.upper() == '8':
            test_encoders_enhanced()
        elif choice.upper() == '9':
            troubleshoot_encoders()
        elif choice.upper() == 'F':
            test_forward_ultrasonic()  # New ultrasonic forward test
        elif choice.upper() == 'R':
            print("Turn test with gyroscope verification...")
            print("Current orientation:", robot_orientation)
            angle = float(input("Enter absolute angle to turn to (0-360): ") or "90")
            print(f"Turning to {angle}° with verification...")
            turn_to_angle_gyro(angle)
            print(f"Turn complete. New orientation: {robot_orientation}°")
        elif choice.upper() == 'G':
            calibrate_gyro_factor()
        elif choice.upper() == 'X':
            toggle_testing_mode()
        elif choice == '0':
            print("Exiting...")
            return False
        else:
            print("Invalid option, please try again")
        return True
    except KeyboardInterrupt:
        emergency_stop()
        return False
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        emergency_stop()
        return False

if __name__ == "__main__":
    try:
        running = True
        while running:
            running = main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    finally:
        stop()
        GPIO.cleanup()
        picam2.stop()
        print("Resources released, program exited safely")