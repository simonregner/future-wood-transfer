
import time

def start_timer():
    return time.time()


def end_timer(timers, start_time):
    end_time = time.time()

    print(f"Callback function executed in {end_time - start_time:.6f} seconds")
    return (int((end_time - start_time) * 10 ** 9) + timers * 4) / 5