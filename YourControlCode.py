import mujoco
import numpy as np


class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, target_points):
    self.m = m
    self.d = d
    self.target_points = target_points
    self.path = self.pathBuilder([10, 5, 20]) #[10, 5, 20]
    self.curr_point = 0
    self.init_qpos = d.qpos.copy()

    # Control gains (using similar values to CircularMotion)
    self.kp = 150.0
    self.kd = 12.0
  
  def pathFinder(self):
    start = self.d.body(mujoco.mj_name2id(self.m, 1, "EE_Frame")).xpos
    points = []
    for i in range(8):
      points.append(self.target_points[:, i])
    distances = {}
    for i in range(8):
      idx = (-1, i)
      distances[idx] = np.linalg.norm(start - points[i])
    
    for i in range(8):
      for j in range(i, 8):
         if (i == j):
            continue
         idx = (i, j)
         distances[idx] = np.linalg.norm(points[i] - points[j])
    
    frontier = PriorityQueue()
    for i in range(i):
       frontier.add([-1, i], distances[(-1, i)])
    
    def growPath(path):
      ps = []
      for i in range(8):
        check = False
        new_p = []
        for j in range(len(path[0])):
          if (i == path[0][j]):
            check = True
          new_p.append(path[0][j])
        if (check):
          continue
        new_p.append(i)
        ordered = tuple(sorted([path[0][-1], i]))
        ps.append([new_p, path[1] + distances[ordered]])
      return ps
      
    while (not frontier.is_empty()):
       curr = frontier.pop()
       if (len(curr[0]) == 9):
         curr[0].pop(0)
         print(curr)
         return curr[0]
       
       new_paths = growPath(curr)
       for i in range(len(new_paths)):
         frontier.add(new_paths[i][0], new_paths[i][1])
    print("uh oh")

  def pathBuilder(self, mid_points):
    path = self.pathFinder()
    path_points = [self.d.body(mujoco.mj_name2id(self.m, 1, "EE_Frame")).xpos] + list(map(lambda x: self.target_points[:, x], path))
    extended_path = []
    if (len(mid_points) == 3):
      mult = mid_points[0]
      adder = mid_points[1]
      max = mid_points[2]
      mid_points = []
      for i in range(8):
        mid_points.append(int(np.rint(mult * np.linalg.norm(path_points[i] - path_points[i - 1]))))
        mid_points[-1] += adder
        if (mid_points[-1] >= max):
          mid_points[-1] = max
        # if (mid_points[-1] <= min):
        #   mid_points[-1] = min
    print(mid_points)
    for i in range(len(path)):
      extended_path.append(path_points[i])
      for j in range(mid_points[i]):
        inter_step = (j + 1) / (mid_points[i] + 1)
        extended_path.append((1 - inter_step) * path_points[i] + inter_step * path_points[i + 1])
    extended_path.append(path_points[-1])
    extended_path.pop(0)
    return extended_path
       

  def CtrlUpdate(self):
    jtorque_cmd = np.zeros(6) #These three lines seem to keep things constant (values like velocity)

    target_position = self.path[self.curr_point]
    
    ee_id = mujoco.mj_name2id(self.m, 1, "EE_Frame")

    jacp = np.zeros((3, 6))
    
    initial_jpos = np.copy(self.d.qpos[:6])
    target_jpos = np.copy(initial_jpos)
    for i in range(3):
      mujoco.mj_jacBodyCom(self.m, self.d, jacp, None, ee_id)
      EE_pos = self.d.body(ee_id).xpos
      pos_err = (target_position - EE_pos)
      dist = np.linalg.norm(pos_err)
      
      target_jpos += (5*dist/4 + 0.85) * np.linalg.pinv(jacp) @ pos_err
      self.d.qpos[:6] = target_jpos
      mujoco.mj_kinematics(self.m, self.d)
    
    self.d.qpos[:6] = np.copy(initial_jpos)
    jpos_error = target_jpos - self.d.qpos[:6]

    velocity = self.d.qvel[:6]
    
    A = np.zeros((6, 6))
    mujoco.mj_fullM(self.m, A, self.d.qM)
    jtorque_cmd += A @ (self.kp * jpos_error - self.kd * velocity) + self.d.qfrc_bias[:6]

    if (np.linalg.norm(EE_pos - target_position) < 0.01):
      self.curr_point += 1
      if (self.curr_point == len(self.path)):
        self.curr_point = 0
      return jtorque_cmd

    return jtorque_cmd
    

class PriorityQueue:
  def __init__(self):
    self.queue = []

  def add(self, item, priority):
    self.queue.append([item, priority])
    self.sift_up(len(self.queue) - 1)

  def pop(self):
    if (len(self.queue) == 0):
      return None
    self.swap(0, len(self.queue) - 1)
    ret = self.queue.pop()
    self.sift_down()
    return ret
  
  def sift_up(self, i):
    curr = i
    par = self.parent(i)
    while (self.queue[par][1] > self.queue[curr][1]):
      self.swap(curr, par)
      curr = par
      par = self.parent(curr)

  def sift_down(self, i = 0):
    size = len(self.queue)
    while True:
        smallest = i
        left = self.left(i)
        right = self.right(i)

        if left < size and self.queue[left][1] < self.queue[smallest][1]:
            smallest = left
        if right < size and self.queue[right][1] < self.queue[smallest][1]:
            smallest = right

        if smallest == i:
            break

        self.swap(i, smallest)
        i = smallest

  def is_empty(self):
    return len(self.queue) == 0

  def parent(self, i):
    return (i - 1) // 2
  
  def left(self, i):
    return 2 * i + 1
  
  def has_left(self, i):
    return self.left(i) < len(self.queue) - 1
  
  def right(self, i):
    return 2 * i + 2
  
  def has_right(self, i):
    return self.right(i) < len(self.queue)
  
  def swap(self, i, j):
    self.queue[i], self.queue[j] = self.queue[j], self.queue[i]