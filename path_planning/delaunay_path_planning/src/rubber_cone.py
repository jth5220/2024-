def find_continuity(triangles):
    triangles = [list(map(int, triangle)) for triangle in triangles]
    result = []
   
    for triangle in triangles:
        temp = [triangle]
        triangles_copy = triangles.copy()
        triangles_copy.remove(triangle)
        
        while True:
            next_triangle = None
            for tri in triangles_copy:
                if len(set(temp[-1]) & set(tri)) == 2:
                    next_triangle = tri
                    break
                    
            if next_triangle is None:
                break
            else:
                temp.append(next_triangle)
                triangles_copy.remove(next_triangle)
                
        result.append(temp)
        
    return result

triangles = [
    [1, 5, 0], [4, 0, 7], [0, 5, 7], [5, 1, 2], [6, 2, 3], [5, 2, 6], 
    [5, 1, 2], [6, 2, 3], [5, 2, 6], [0, 1, 8], [8, 1, 5], [4, 0, 8], [8, 7, 4],
    [6, 9, 2], [4, 9, 0], [0, 9, 6], [7, 2, 3], [6, 2, 7], [6, 8, 1], [1, 0, 6], [1, 8, 5],
    [4, 1, 5], [0, 5, 1], [4, 5, 6], [3, 6, 2], [5, 0, 2], [2, 6, 5],
    [0, 4, 3], [4, 0, 1], [1, 5, 4], [2, 5, 1],
    [2, 5, 1], [1, 5, 0], [3, 6, 4], [0, 5, 4], [4, 6, 0],
    [4, 0, 5], [6, 5, 1], [5, 0, 1], [2, 6, 1], [7, 4, 3], [0, 4, 7]
]
