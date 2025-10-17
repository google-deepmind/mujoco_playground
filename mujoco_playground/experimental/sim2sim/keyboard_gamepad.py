import threading
import time
import sys
import os

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
    import select  # Import select


def _interpolate(value, old_max, new_scale, deadzone=0.01):
    """Interpolates a value, applying a deadzone."""
    ret = value * new_scale / old_max
    if abs(ret) < deadzone:
        return 0.0
    return ret


class KeyboardGamepad:
    """
    Emulates a gamepad using keyboard input (WASD and QE).

    Attributes:
        vx (float): Velocity in the x-direction.
        vy (float): Velocity in the y-direction.
        wz (float): Rotational velocity.
        is_running (bool): Flag indicating if the input loop is running.
        vel_scale_x (float): Scaling factor for x-velocity.
        vel_scale_y (float): Scaling factor for y-velocity.
        vel_scale_rot (float): Scaling factor for rotational velocity.
    """

    def __init__(
        self,
        vel_scale_x=0.4,
        vel_scale_y=0.4,
        vel_scale_rot=1.0,
        acceleration=0.1,  # Added acceleration
        max_speed=5.0,  # Added maximum speed
    ):
        """
        Initializes the KeyboardGamepad.

        Args:
            vel_scale_x (float): Scaling factor for x-velocity.
            vel_scale_y (float): Scaling factor for y-velocity.
            vel_scale_rot (float): Scaling factor for rotational velocity.
            acceleration (float): Rate at which velocity changes per key press.
            max_speed (float): Maximum allowed speed in any direction.
        """
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.is_running = True
        self.vel_scale_x = vel_scale_x
        self.vel_scale_y = vel_scale_y
        self.vel_scale_rot = vel_scale_rot
        self.acceleration = acceleration
        self.max_speed = max_speed

        self._key_mapping = {  # Maps keys to velocity changes
            'w': (0, acceleration, 0),
            's': (0, -acceleration, 0),
            'a': (acceleration, 0, 0),
            'd': (-acceleration, 0, 0),
            'q': (0, 0, acceleration),
            'e': (0, 0, -acceleration),
            ' ': (0, 0, 0),  # Space key for resetting velocity
        }

        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    def read_loop(self):
        """
        Reads keyboard input and updates velocities.
        """
        if os.name == 'nt':
            # Windows
            while self.is_running:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    self.update_command(key)
                time.sleep(0.01)  # Reduce CPU usage
        else:
            # Linux and macOS
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                while self.is_running:
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1).lower()
                        self.update_command(key)
                    time.sleep(0.01)  # Reduce CPU usage
            except Exception as e:
                print(f"Error reading input: {e}")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def update_command(self, key):
        """
        Updates velocities based on the pressed key.

        Args:
            key (str): The pressed key.
        """
        if key in self._key_mapping:
            dx, dy, dw = self._key_mapping[key]

            # Handle space bar for resetting velocity
            if key == ' ':
                self.vx = 0.0
                self.vy = 0.0
                self.wz = 0.0
                print(f"Velocity reset to zero at {time.time()}")
            else:
                # Update velocities
                self.vx += dx
                self.vy += dy
                self.wz += dw

                # Clamp velocities to max_speed
                self.vx = max(-self.max_speed, min(self.vx, self.max_speed))
                self.vy = max(-self.max_speed, min(self.vy, self.max_speed))
                self.wz = max(-self.max_speed, min(self.wz, self.max_speed))
            print(f"Velocity changed to ({self.vx:.2f}, {self.vy:.2f}, {self.wz:.2f}) at {time.time()}")

    def get_command(self):
        """
        Returns the current velocity command.

        Returns:
            numpy.ndarray: An array containing vx, vy, and wz.
        """
        # Swapped x and y axes here:
        return [
            _interpolate(self.vy, self.max_speed, self.vel_scale_y),  # y becomes first
            _interpolate(self.vx, self.max_speed, self.vel_scale_x),  # x becomes second
            _interpolate(self.wz, self.max_speed, self.vel_scale_rot),
        ]

    def stop(self):
        """
        Stops the input loop.
        """
        self.is_running = False


if __name__ == "__main__":
    import select  # Imported here to avoid issues if select isn't available

    gamepad = KeyboardGamepad()
    print("Use WASD to control linear velocity, QE to control rotational velocity.")
    print("Press Spacebar to reset velocity to zero.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            command = gamepad.get_command()
            print(f"Command: {command}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
        gamepad.stop()
        time.sleep(0.5)  # Give the thread time to stop
