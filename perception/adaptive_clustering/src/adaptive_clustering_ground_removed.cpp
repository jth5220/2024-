// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "adaptive_clustering/ClusterArray.h"
#include <math.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

//#define LOG

ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher marker_array_pub_;

//ros::Publisher marker_pose_pub_;

bool print_fps_;
float z_axis_min_;
float z_axis_max_;
double PI = 3.1415926;
int cluster_size_min_;
int cluster_size_max_;

float size_x;
float size_y;
float size_z;
// float center_x;
// float center_y;
// float center_z;

const int region_max_ = 10; // Change this value to match how far you want to detect. 단위가 뭐냐 
int regions_[100];

int frames; clock_t start_time; bool reset = true; //fps

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) {

  if(print_fps_)if(reset){frames=0;start_time=clock();reset=false;}//fps
  
  /*** Convert ROS message to PCL ***/
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);
  
  /*** Remove ground and ceiling ***/
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  for (size_t i = 0; i < pcl_pc_in->points.size(); ++i) {
    pc_indices->push_back(i);
  }
  // pcl::PassThrough<pcl::PointXYZ> pt;

  // pt.setInputCloud(pcl_pc_in);
  // pt.setFilterFieldName("z");
  // pt.setFilterLimits(-0.65, 0.5); // 0.2
  // pt.filter(*pcl_pc_in);
  
  // pt.setInputCloud(pcl_pc_in);
  // pt.setFilterFieldName("y");
  // pt.setFilterLimits(-3., 3.);
  // pt.filter(*pcl_pc_in);

  // pt.setInputCloud(pcl_pc_in);
  // pt.setFilterFieldName("x");
  // pt.setFilterLimits(-1, 6);
  // pt.filter(*pc_indices);

  /*** Divide the point cloud into nested circular regions ***/
  boost::array<std::vector<int>, region_max_> indices_array;
  for(int i = 0; i < pc_indices->size(); i++) {
    float range = 0.0;
    for(int j = 0; j < region_max_; j++) {
      float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
	pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
	pcl_pc_in->points[(*pc_indices)[i]].z * pcl_pc_in->points[(*pc_indices)[i]].z;
      if(d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
      	indices_array[j].push_back((*pc_indices)[i]);
      	break;
      }
      range += regions_[j];
    }
  }
  
  /*** Euclidean clustering ***/
  float tolerance = 0.0;
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZ>::Ptr > > clusters;
  
  for(int i = 0; i < region_max_; i++) {
    tolerance += 0.1;
    if(indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(pcl_pc_in, indices_array_ptr);
      
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_in);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);
      
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
      	pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      	for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
      	  cluster->points.push_back(pcl_pc_in->points[*pit]);
  	}
      	cluster->width = cluster->size();
      	cluster->height = 1;
      	cluster->is_dense = true;
	clusters.push_back(cluster);
      }
    }
  }
  
  /*** Output ***/
  if(cloud_filtered_pub_.getNumSubscribers() > 0) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZ>);
    sensor_msgs::PointCloud2 ros_pc2_out;
    pcl::copyPointCloud(*pcl_pc_in, *pc_indices, *pcl_pc_out);
    pcl::toROSMsg(*pcl_pc_out, ros_pc2_out);
    cloud_filtered_pub_.publish(ros_pc2_out);
  }
  
  adaptive_clustering::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::MarkerArray marker_pose;

  for(int i = 0; i < clusters.size(); i++) 
  {
    if(cluster_array_pub_.getNumSubscribers() > 0) 
    {
      sensor_msgs::PointCloud2 ros_pc2_out;
      pcl::toROSMsg(*clusters[i], ros_pc2_out);
      cluster_array.clusters.push_back(ros_pc2_out);
    }
    
    if(marker_array_pub_.getNumSubscribers() > 0) 
    {
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);
      
      visualization_msgs::Marker marker;
      
      marker.header = ros_pc2_in->header;
      marker.ns = "adaptive_clustering";
      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_LIST;
      
      geometry_msgs::Point p[24];
      p[0].x = max[0];  p[0].y = max[1];  p[0].z = max[2];
      p[1].x = min[0];  p[1].y = max[1];  p[1].z = max[2];
      p[2].x = max[0];  p[2].y = max[1];  p[2].z = max[2];
      p[3].x = max[0];  p[3].y = min[1];  p[3].z = max[2];
      p[4].x = max[0];  p[4].y = max[1];  p[4].z = max[2];
      p[5].x = max[0];  p[5].y = max[1];  p[5].z = min[2];
      p[6].x = min[0];  p[6].y = min[1];  p[6].z = min[2];
      p[7].x = max[0];  p[7].y = min[1];  p[7].z = min[2];
      p[8].x = min[0];  p[8].y = min[1];  p[8].z = min[2];
      p[9].x = min[0];  p[9].y = max[1];  p[9].z = min[2];
      p[10].x = min[0]; p[10].y = min[1]; p[10].z = min[2];
      p[11].x = min[0]; p[11].y = min[1]; p[11].z = max[2];
      p[12].x = min[0]; p[12].y = max[1]; p[12].z = max[2];
      p[13].x = min[0]; p[13].y = max[1]; p[13].z = min[2];
      p[14].x = min[0]; p[14].y = max[1]; p[14].z = max[2];
      p[15].x = min[0]; p[15].y = min[1]; p[15].z = max[2];
      p[16].x = max[0]; p[16].y = min[1]; p[16].z = max[2];
      p[17].x = max[0]; p[17].y = min[1]; p[17].z = min[2];
      p[18].x = max[0]; p[18].y = min[1]; p[18].z = max[2];
      p[19].x = min[0]; p[19].y = min[1]; p[19].z = max[2];
      p[20].x = max[0]; p[20].y = max[1]; p[20].z = min[2];
      p[21].x = min[0]; p[21].y = max[1]; p[21].z = min[2];
      p[22].x = max[0]; p[22].y = max[1]; p[22].z = min[2];
      p[23].x = max[0]; p[23].y = min[1]; p[23].z = min[2];
      for(int i = 0; i < 24; i++) {
  	marker.points.push_back(p[i]);
      }
      
      size_x = std::abs(max[0]-min[0]);
      size_y = std::abs(max[1]-min[1]);
      size_z = std::abs(max[2]-min[2]);
      
      // center_x = (max[0]+min[0])/2;
      // center_y = (max[1]+min[1])/2;
      // center_z = (max[2]+min[2])/2;

      marker.scale.x = 0.02;
      marker.color.a = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.5;

      marker.lifetime = ros::Duration(0.1);

      if (size_z > 0.2 && size_x <1 && size_y <1 && size_z <1.5) 
      {
        marker_array.markers.push_back(marker);
      }
      // marker_array.markers.push_back(marker);
      
      marker.header.stamp = ros::Time::now();
      // marker.header.stamp = ros_pc2_in->header;
      marker.ns = "clustering_pose";
      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_LIST;

      marker.points.clear();
      marker.scale.x = std::abs(max[0]-min[0]);
      marker.scale.y = std::abs(max[1]-min[1]);
      marker.scale.z = std::abs(max[2]-min[2]);
      marker.pose.position.x = (max[0]+min[0])/2;
      marker.pose.position.y = (max[1]+min[1])/2;
      marker.pose.position.z = (max[2]+min[2])/2;
      marker.color.a = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.5;
      marker.lifetime = ros::Duration(0.1);

      if ( size_z > 0.2 && size_x <1 && size_y <1 && size_z <1.5 ) // 라바콘을 잡기 위한 파라미터
      {
        marker_pose.markers.push_back(marker);
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*clusters[i], centroid);
        
        geometry_msgs::Pose pose;
        pose.position.x = centroid[0];
        pose.position.y = centroid[1];
        pose.position.z = centroid[2];
        pose.orientation.x = std::abs(max[0]-min[0]);
        pose.orientation.y = std::abs(max[1]-min[1]);
        pose.orientation.z = std::abs(max[2]-min[2]);
        pose.orientation.w = 1;
        pose_array.poses.push_back(pose);
      }

      if(pose_array_pub_.getNumSubscribers() > 0) 
      {

      }
    }

  }
  
  if(cluster_array.clusters.size()) {
    cluster_array.header = ros_pc2_in->header;
    cluster_array_pub_.publish(cluster_array);
  }

  if(pose_array.poses.size()) {
    pose_array.header = ros_pc2_in->header;
    pose_array_pub_.publish(pose_array);
  }
  
  if(marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }
  
  if(print_fps_)if(++frames>10){std::cerr<<"[adaptive_clustering] fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;}//fps

}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_clustering");
  
  /*** Subscribers ***/
  ros::NodeHandle nh;
  ros::Subscriber point_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("velodyne_points_ground_removed", 1, pointCloudCallback);

  /*** Publishers ***/
  ros::NodeHandle private_nh("~");
  cluster_array_pub_ = private_nh.advertise<adaptive_clustering::ClusterArray>("clusters", 100);
  // cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);

  //marker_pose_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers_pose", 100);

  /*** Parameters ***/
  std::string sensor_model;
  
  private_nh.param<std::string>("sensor_model", sensor_model, "VLP-16"); // VLP-16, HDL-32E, HDL-64E
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<float>("z_axis_min", z_axis_min_, -0.5);
  private_nh.param<float>("z_axis_max", z_axis_max_, 1.0); // 0.1
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 3);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 10000);
  
  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper "Online learning for human classification in 3D LiDAR-based tracking"

  if(sensor_model.compare("VLP-16") == 0) {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    ROS_FATAL("Unknown sensor model!");
  }
  
  ros::spin();

  return 0;
}
