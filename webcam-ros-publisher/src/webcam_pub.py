#!/usr/bin/env python3
import os
import time
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def env_str(name, default):
    return os.environ.get(name, default)

def env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def main():
    # Load config
    device_id = env_int("CAM_DEVICE", 0)
    topic = env_str("ROS_IMAGE_TOPIC", "/camera/image_raw")
    target_pub_fps = env_float("CAM_FPS", 15.0)
    frame_id = env_str("FRAME_ID", "camera")
    
    # Camera hardware requests
    req_w = env_int("CAM_WIDTH", 640)
    req_h = env_int("CAM_HEIGHT", 480)
    req_hw_fps = env_float("CAM_CAPTURE_FPS", 30.0) # Request fast HW capture
    
    debug_every = env_int("DEBUG_EVERY_N", 0)

    # 1. Initialize Node with anonymous=True to prevent name collisions
    rospy.init_node("webcam_publisher", anonymous=True)
    
    # Publisher & Bridge
    pub = rospy.Publisher(topic, Image, queue_size=1)
    bridge = CvBridge()

    rospy.loginfo(f"Opening /dev/video{device_id}...")
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

    if not cap.isOpened():
        rospy.logerr(f"Could not open video device {device_id}")
        return

    # 2. Configure Camera
    # It is better to request a high HW FPS (e.g. 30) and throttle via software
    # to ensure the auto-exposure/white-balance algorithms work smoothly.
    if req_w > 0 and req_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    if req_hw_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, req_hw_fps)
        
    # Attempt to minimize buffer size to reduce lag (Linux/V4L2 backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Read actuals
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    rospy.loginfo(f"Camera Info: {actual_w}x{actual_h} @ {actual_fps}FPS (Hardware)")
    rospy.loginfo(f"Publishing to: {topic} @ Target {target_pub_fps}FPS")

    # 3. The 'Always-Read' Loop
    # We calculate the interval between publishes manually
    pub_interval = 1.0 / target_pub_fps
    last_pub_time = 0
    frame_count = 0
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        
        if not ret:
            rospy.logwarn("Frame read failed")
            time.sleep(0.1)
            continue

        now = time.time()
        
        # Check if it is time to publish
        if (now - last_pub_time) >= pub_interval:
            try:
                msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = frame_id
                pub.publish(msg)
                
                last_pub_time = now
                frame_count += 1
                
                if debug_every > 0 and (frame_count % debug_every == 0):
                    rospy.loginfo(f"Pub Stats: Frame {frame_count}")
                    
            except Exception as e:
                rospy.logerr(f"Publish error: {e}")

        # Small sleep to prevent 100% CPU usage if camera is extremely fast
        # but keep it very short to ensure we don't miss clearing the buffer.
        time.sleep(0.001)

    cap.release()
    rospy.loginfo("Webcam publisher stopped.")

if __name__ == "__main__":
    main()