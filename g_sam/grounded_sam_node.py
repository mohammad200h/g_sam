#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from PIL import Image as Img
import numpy as np


from .grounded_sam import grounded_sam
from foundation_pose_interfaces.srv import Registration
from sensor_msgs.msg import  CameraInfo



class GSAMNode(Node):
  def __init__(self):
    super().__init__('Grounded_SAM')
    self.get_logger().info('Grounded SAM is Up! Lets get Segmenting')

    self.color = None
    self.depth = None
    self.g_sam_msg = None
    self.camera_info = None

    # Services
    self.regeneration_service =  self.create_service(Registration,
                                                    '/foundation_pose_regeneration',
                                                    self.handle_registration)


    # Subscribers
    self.subscription = self.create_subscription(
          Image,
          '/camera/camera/color/image_raw',
          self.grab_color,
          10
    )
    self.camera_info = self.create_subscription(
          CameraInfo,
          '/camera/camera/depth/camera_info',
          self.get_camera_info,
          10
    )
    self.sub_depth = self.create_subscription(
          Image,
          '/camera/camera/aligned_depth_to_color/image_raw',
          self.grab_depth, 10
    )
    # Publisher
    self.gsam_mask_publisher = self.create_publisher(Image,
                                                '/g_sam/mask',
                                                 10)
    self.gsam_color_publisher = self.create_publisher(Image,
                                                '/g_sam/color',
                                                 10)


    # timer callbacks
    self.frame_rate = 60 # FPS
    self._timer = self.create_timer(1.0 / self.frame_rate, self.timer_callback)
  def timer_callback(self):
    if self.g_sam_msg:
      self.gsam_mask_publisher.publish(self.g_sam_msg["mask"])
      self.gsam_color_publisher.publish(self.g_sam_msg["color"])


  def grab_depth(self, msg):
    depth = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")/1e3

    depth[(depth<0.001) | (depth>=np.inf)] = 0
    msg =  CvBridge().cv2_to_imgmsg(depth, encoding="passthrough")

    self.depth = msg

  def grab_color(self, msg):
    self.color = msg

  def get_camera_info(self,msg):
    self.camera_info = msg


  def get_mask(self,prompt:str):
    bridge = CvBridge()
    image_cv = bridge.imgmsg_to_cv2(self.color,  desired_encoding="bgr8")
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # process image to get mask
    masks = grounded_sam(image_rgb,prompt)
    mask_array = masks.cpu().numpy()[0][0]  # Get the first mask

    print(f"mask_array::{mask_array}")

    # Save the image
    mask_image  =Img.fromarray(mask_array)
    mask_image.save("mask.png")

    # Save the image
    image_rgb  =Img.fromarray(image_rgb)
    image_rgb.save("color.png")
    print(f"Before normalization: min={mask_array.min()}, max={mask_array.max()}, dtype={mask_array.dtype}")
    mask_array = (mask_array * 255).astype('uint8')

    print(f"grounded_sam::type::{type(mask_array)}")
    print(f"grounded_sam::shape::{mask_array.shape}")
    print(f"grounded_sam::dtype::{mask_array.dtype}")


    msg =  bridge.cv2_to_imgmsg(mask_array, encoding="passthrough")

    self.g_sam_msg = {
      "mask":msg,
      "color":self.color
    }

    return msg

   # Service callback
  def handle_registration(self,request,response):
    while not (self.color and self.depth and self.camera_info):
      waiting_for = None
      if self.color == None:
        waiting_for = "color"
      elif self.depth == None:
        waiting_for = "depth"
      elif self.camera_info == None:
        waiting_for = "camera_info"

      self.get_logger().info(f'waiting for {waiting_for}')

    response.color = self.color
    response.depth = self.depth
    response.camera_info = self.camera_info

    # get mask
    prompt = request.prompt.data
    print(f"prompt::{prompt}")
    response.mask = self.get_mask(request.prompt.data)

    return response


def main(args=None):
    rclpy.init(args=args)
    node = GSAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
