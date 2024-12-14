#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as Img
import numpy as np

from .grounded_sam import grounded_sam


class MyNode(Node):
  def __init__(self):
    super().__init__('Grounded_SAM')
    self.get_logger().info('Grounded SAM is Up! Lets get Segmenting')
    # Create a subscriber
    self.subscription = self.create_subscription(
          Image,
          '/camera/camera/color/image_raw',
          self.image_callback,
          10
    )
    self.subscription  # prevent unused variable warning
    # Create a publisher
    self.publisher = self.create_publisher(
          Image,      # Message type
          '/mask_image',  # Topic name
          10          # QoS (queue size)
    )
    self.img_num = 0

  def image_callback(self, msg):
    self.get_logger().info('Received an image!')
    bridge = CvBridge()
    image_cv = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    if self.img_num == 0:
      self.img_num = self.img_num + 1
      ### RUN GROUNDED SAM ###
      masks = grounded_sam(image_cv)
      #### Pub Mask Image ###
      self.publish_mask_image(masks)

  def publish_mask_image(self,masks):
    bridge = CvBridge()
    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Convert mask to binary image
        mask_img = Img.fromarray(mask_np[0])  # Convert numpy array to PIL Image
        mask_img_cv = np.array(mask_img)  # Convert PIL Image to numpy array
        mask_msg = bridge.cv2_to_imgmsg(mask_img_cv, encoding="mono8")  # Convert OpenCV image to ROS Image message
        self.publisher.publish(mask_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
