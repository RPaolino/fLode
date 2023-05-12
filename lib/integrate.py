import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################
# EXPLICIT METHODS
###############################################
forward_euler = {
  "c": torch.tensor([0], device=device),
  "b": torch.tensor([1], device=device),
}
explicit_midpoint = {
  "c": torch.tensor([0, 1/2], device=device),
  "b": torch.tensor([0, 1], device=device),
  "A": torch.tensor([
      [0],
      [1/2]
    ], device=device)
}
heun2 = {
  "c": torch.tensor([0, 1], device=device),
  "b": torch.tensor([1/2, 1/2], device=device),
  "A": torch.tensor([
      [0],
      [1]
    ], device=device)
}
ralston2 = {
  "c": torch.tensor([0, 2/3], device=device),
  "b": torch.tensor([1/4, 3/4], device=device),
  "A": torch.tensor([
      [0],
      [2/3]
    ], device=device)
}

rk3 = {
  "c": torch.tensor([0, 1/2, 1], device=device),
  "b": torch.tensor([1/6, 2/3, 1/6], device=device),
  "A": torch.tensor([
      [0, 0],
      [1/2, 0],
      [-1, 2]
    ], device=device)
}

heun3 = {
  "c": torch.tensor([0, 1/3, 2/3], device=device),
  "b": torch.tensor([1/4, 0, 3/4], device=device),
  "A": torch.tensor([
      [0, 0],
      [1/3, 0],
      [0, 2/3]
    ], device=device)
}

wray3 = {
  "c": torch.tensor([0, 8/15, 2/3], device=device),
  "b": torch.tensor([1/4, 0, 3/4], device=device),
  "A": torch.tensor([
      [0, 0],
      [8/15, 0],
      [1/4, 5/12]
    ], device=device)
}

ralston3 = {
  "c": torch.tensor([0, 1/2, 3/4], device=device),
  "b": torch.tensor([2/9, 1/3, 4/9], device=device),
  "A": torch.tensor([
      [0, 0],
      [1/2, 0],
      [0, 3/4]
    ], device=device)
}

ssprk3 = {
  "c": torch.tensor([0, 1, 1/2], device=device),
  "b": torch.tensor([1/6, 1/6, 2/3], device=device),
  "A": torch.tensor([
      [0, 0],
      [1, 0],
      [1/4, 1/4]
    ], device=device)
}
rk4 = {
  "c": torch.tensor([0, 1/2, 1/2, 1], device=device),
  "b": torch.tensor([1/6, 1/3, 1/3, 1/6], device=device),
  "A": torch.tensor([
      [0, 0, 0],
      [1/2, 0, 0],
      [0, 1/2, 0],
      [0, 0, 1]
    ], device=device)}
rk375 = {
  "c": torch.tensor([0, 1/3, 2/3, 1], device=device),
  "b": torch.tensor([1/8, 3/8, 3/8, 1/8], device=device),
  "A": torch.tensor([
      [0, 0, 0],
      [1/3, 0, 0],
      [-1/3, 1, 0],
      [1, -1, 1]
    ], device=device)}
ralston4 = {
  "c": torch.tensor([0, .4, .45573725 , 1], device=device),
  "b": torch.tensor([.17476028, -.55148066, 1.20553560, .17118478], device=device),
  "A": torch.tensor([
      [0, 0, 0],
      [.4, 0, 0],
      [.29697761, .15875964, 0],
      [.21810040, -3.05096516, 3.83286476]
    ], device=device)
}
###############################################
# ADAPTIVE/EMBEDDED
###############################################

heun_euler = {
  "c": torch.tensor([0, 1], device=device),
  "b": torch.tensor([1/2, 1/2], device=device),
  "b*": torch.tensor([1, 0], device=device),
  "A": torch.tensor([
      [0],
      [1]
    ], device=device)
}

fehlberg_rk1 = {
  "c": torch.tensor([0, 1/2, 1], device=device),
  "b": torch.tensor([1/512, 255/256, 1/512], device=device),
  "b*": torch.tensor([1/256, 255/256, 0], device=device),
  "A": torch.tensor([
        [0, 0],
        [1/2, 0], 
        [1/256, 255/256]
    ], device=device)
}

bogacki_shampine = {
  "c": torch.tensor([0, 1/2, 3/4, 1], device=device),
  "b": torch.tensor([2/9, 1/3, 4/9, 0], device=device),
  "b*": torch.tensor([7/24, 1/4, 1/3, 1/8], device=device),
  "A": torch.tensor([
        [0, 0, 0],
        [1/2, 0, 0], 
        [0, 3/4, 0],
        [2/9, 1/3, 4/9],
    ], device=device)
}

fehlberg = {
  "c": torch.tensor([0, 1/4, 3/8, 12/13, 1, 1/2], device=device),
  "b": torch.tensor([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55], device=device),
  "b*": torch.tensor([25/216, 0, 1408/2565, 2197/4104, -1/5, 0], device=device),
  "A": torch.tensor([
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0], 
        [3/32, 9/32, 0, 0, 0],
        [1932/2197, -7200/2197,7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40],
    ], device=device)
}

cash_karp = {
  "c": torch.tensor([0, 1/5, 3/10, 3/5, 1, 7/8], device=device),
  "b": torch.tensor([37/378, 0, 250/621, 125/594, 0, 512/1771], device=device),
  "b*": torch.tensor([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4], device=device),
  "A": torch.tensor([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0], 
        [3/40, 9/40, 0, 0, 0],
        [3/10, -9/10, 6/5, 0, 0],
        [-11/54, 5/2, -70/27, 35/27, 0],
        [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
    ], device=device)
}

dopri5 = {
  "c": torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1], device=device),
  "b": torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], device=device),
  "b*": torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], device=device),
  "A": torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0], 
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]], device=device)
}