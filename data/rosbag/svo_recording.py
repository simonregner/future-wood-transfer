import pyzed.sl as sl
import argparse

def main():
    # Parse command line arguments for the recording filename
    parser = argparse.ArgumentParser(description="Record ZED2 stream in HD1080 resolution with specified filename.")
    parser.add_argument("filename", type=str, help="The output filename for the recording (e.g., recording_hd1080.svo)")
    args = parser.parse_args()
    recording_filename = args.filename

    # Create a ZED camera object
    zed = sl.Camera()

    # Initialize the parameters for the camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution to HD1080
    init_params.camera_fps = 30                           # Optionally set the frame rate

    # Open the camera with the initialization parameters
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error opening ZED camera: ", err)
        return

    # Set up the recording parameters (using H264 compression mode)
    recording_params = sl.RecordingParameters(recording_filename, sl.SVO_COMPRESSION_MODE.H264)

    # Start recording
    if zed.enable_recording(recording_params) != sl.ERROR_CODE.SUCCESS:
        print("Error: Unable to start recording")
        zed.close()
        return

    print(f"Recording started. Saving to '{recording_filename}'. Press Ctrl+C to stop.")

    try:
        # Main recording loop
        while True:
            # Grab a new frame. If successful, the frame is automatically added to the recording.
            err = zed.grab()
            if err != sl.ERROR_CODE.SUCCESS:
                print("Grab error: ", err)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        # Clean up: disable recording and close the camera
        zed.disable_recording()
        zed.close()

if __name__ == "__main__":
    main()
