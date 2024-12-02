#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.tri as mtri
from collections import OrderedDict


class DelaunayPathPlanning():
    def __init__(self):
        self.cones = None
        self.labels = None
        self.prev_midpoints = None
        return
        
    def update(self, cones, labels):       
        self.cones = np.array(cones)
        self.labels = labels
         
        # 라바콘이 3개 이하로 들어오면
        if len(self.labels) < 3:
            return self.prev_midpoints
        
        delaunay_triangles = mtri.Triangulation(self.cones[:,0], self.cones[:,1])
        triangles = np.array(delaunay_triangles.get_masked_triangles(), dtype=np.uint8) # [[a,b,c], [d,e,f], ...]
        
        triangles_inlier = self.get_inlier_triangle(triangles)
        
        # 모두 같은 색 라바콘인 경우에는
        if len(triangles_inlier) == 0 :
            return self.prev_midpoints
        
        midpoints = self.find_path(triangles_inlier)
        self.prev_midpoints = midpoints
        
        return midpoints
    
    def get_inlier_triangle(self, triangles):
        deltri_inlier = []
        for triangle in triangles:
            if not self._is_same_color(triangle):
                deltri_inlier.append(triangle)
                
        return deltri_inlier
    
    # def get_midpoints(self, path):
    #     midpoints = np.empty((0,2))

    #     prev_tri = None
    #     for i, triangle in enumerate(path):
    #         if i == 0:
    #             a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
    #             a_color, a_xy = self.labels[a_idx], self.cones[a_idx]
    #             b_color, b_xy = self.labels[b_idx], self.cones[b_idx]
    #             c_color, c_xy = self.labels[c_idx], self.cones[c_idx]
                
    #             # midpoint 계산
    #             if(a_color == b_color):
    #                 midpoint_1 = (b_xy + c_xy)/2
    #                 midpoint_2 = (c_xy + a_xy)/2
    #                 if midpoint_1[0] < midpoint_2[0]:
    #                     midpoints = np.vstack((midpoints, [midpoint_1, midpoint_2]))
    #                 else:
    #                     midpoints = np.vstack((midpoints, [midpoint_2, midpoint_1]))
                        
    #             elif(b_color == c_color):
    #                 midpoint_1 = (a_xy + b_xy)/2
    #                 midpoint_2 = (c_xy + a_xy)/2
    #                 if midpoint_1[0] < midpoint_2[0]:
    #                     midpoints = np.vstack((midpoints, [midpoint_1, midpoint_2]))
    #                 else:
    #                     midpoints = np.vstack((midpoints, [midpoint_2, midpoint_1]))
                    
    #             elif(c_color == a_color): 
    #                 midpoint_1 = (a_xy + b_xy)/2
    #                 midpoint_2 = (b_xy + c_xy)/2
    #                 if midpoint_1[0] < midpoint_2[0]:
    #                     midpoints = np.vstack((midpoints, [midpoint_1, midpoint_2]))
    #                 else:
    #                     midpoints = np.vstack((midpoints, [midpoint_2, midpoint_1]))
                    
    #         else:
    #             common_indicies = list(set(triangle) & set(prev_tri))
    #             noncommon_index = [e for e in triangle if e not in common_indicies][0]
    #             # print(self.labels)
    #             # print("noncommon_index: ",self.labels[noncommon_index], noncommon_index)
    #             if self.labels[noncommon_index] != self.labels[common_indicies[0]]:
    #                 xy_1 = self.cones[noncommon_index]
    #                 xy_2 = self.cones[common_indicies[0]]
                    
    #                 midpoints = np.vstack((midpoints, [(xy_1 + xy_2)/2]))
    #             else:
    #                 xy_1 = self.cones[noncommon_index]
    #                 xy_2 = self.cones[common_indicies[-1]]
                    
    #                 midpoints = np.vstack((midpoints, [(xy_1 + xy_2)/2]))
    #         prev_tri = triangle
                
    #     return midpoints
    
    def find_path(self, triangles):
        graph = {tuple(sorted(triangle)): [] for triangle in triangles}
        for triangle in triangles:
            for other_triangle in triangles:
                if triangle is not other_triangle and len(set(triangle) & set(other_triangle)) == 2:
                    graph[tuple(sorted(triangle))].append(tuple(sorted(other_triangle)))
        
        # 시작 삼각형 찾기
        first_tri = None
        dist_min = np.inf
        target_point = np.array([-2., 0.])
        for tri in graph.keys():
            if len(graph[tri]) == 1:
                dist = self._get_triangle_distance(tri, target_point)
                if dist < dist_min:
                    dist_min = dist
                    first_tri = tri        
        
        if first_tri is None:
            return self.prev_midpoints
        
        # graph 구조를 통해 삼각형 이어주기
        path_cones=[]
        path = [first_tri]
        outlier = []
        cur_node = first_tri
        for i in range(len(graph.keys())):
            neighbors = graph[cur_node]
            for neighbor in neighbors:
                if neighbor not in path and neighbor not in outlier:
                    common_indicies = list(set(cur_node) & set(neighbor))
                    try:
                        if self.labels[common_indicies[0]] != self.labels[common_indicies[1]]:
                            path.append(neighbor)
                            cur_node = neighbor
                            path_cones.append([common_indicies[0], common_indicies[1]])
                        else: 
                            outlier.append(neighbor)
                    
                    except:
                        print('common_indicies: ', common_indicies)
            #     else:
            #         cnt += 1
            # if cnt > 2:
            #     break
            
        # 첫 번째 삼각형에서 미추가된 라바콘 세트 추가
        common_indicies = list(set(first_tri) & set(path_cones[0]))
        noncommon_index = [e for e in first_tri if e not in common_indicies][0]
        if self.labels[common_indicies[0]] != self.labels[noncommon_index]:
            path_cones.insert(0, [common_indicies[0], noncommon_index])
        else:
            path_cones.insert(0, [common_indicies[1], noncommon_index])
        
        # 마지막 삼각형에서 미추가된 라바콘 세트 추가
        last_tri = path[-1]
        common_indicies = list(set(last_tri) & set(path[-2]))
        noncommon_index = [e for e in last_tri if e not in common_indicies][0]
        if self.labels[common_indicies[0]] != self.labels[noncommon_index]:
            path_cones.append([common_indicies[0], noncommon_index])
        else:
            path_cones.append([common_indicies[1], noncommon_index])
        # print(path_cones)
        
        # midpoint 계산
        midpoints = np.empty((0,2))
        for cone_1, cone_2 in path_cones:
            midpoint = (self.cones[cone_1] + self.cones[cone_2])/2
            midpoints = np.vstack((midpoints, [midpoint]))
        # path = []  # 방문한 노드를 저장하기 위한 리스트
        # stack = [first_tri]  # 탐색을 위한 스택, 시작 노드로 초기화
        # while stack:  # 스택이 빌 때까지
        #     node = stack.pop()  # 스택의 마지막 요소를 꺼냄
        #     if node not in path:  # 꺼낸 노드가 방문한 적 없는 노드라면
        #         path.append(node)  # 방문 리스트에 추가
        #         neighbors = graph[node]  # 해당 노드의 이웃 노드들을 가져옴
        #         for neighbor in neighbors:  # 이웃 노드들 중에서
        #             if neighbor not in path:  # 방문하지 않은 노드를
        #                 stack.append(neighbor)  # 스택에 추가
        # print(path)
        
        # path = [first_tri]
        # cur_node = first_tri
        # for i in range(len(graph.keys())):
        #     neighbors = graph[cur_node]
        #     if len(neighbors)>=3:
        #         print("3개 이상")
        #         print(graph)
        #     for neighbor in neighbors:
        #         if neighbor not in path:
        #             path.append(neighbor)
        #             cur_node = neighbor
                    
        # print(graph)
        # return path  # 방문한 노드의 순서대로 반환
        return midpoints
            
    
    def _get_triangle_distance(self, triangle, target_point):
        a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
        centroid = (self.cones[a_idx] + self.cones[b_idx] + self.cones[c_idx]) / 3
        distance = np.linalg.norm(centroid - target_point)
        return distance
    
    def _is_same_color(self, triangle):
        a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
        
        if self.labels[a_idx] == self.labels[b_idx] == self.labels[c_idx]:
                return True
        return False
    
    def _get_triangle_color(self, triangle):
        a_idx, b_idx, c_idx = triangle[0], triangle[1], triangle[2]
        return self.labels[a_idx], self.labels[b_idx], self.labels[c_idx]
    
def DFS(graph, start):
    cnt = 0
    path=[]
    stack = []
    stack.append(start)
    
    while stack:
        if cnt >= 30:
            break
        n = stack.pop()
        if n not in path:
            path.append(n)
            for i in graph[n]:
                if i not in path:
                    stack.append(i)
        cnt += 1
    return path