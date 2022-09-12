import numpy as np

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9, Eg=None):
        self.learning_rate = learning_rate
        self.Eg = Eg # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_wrt_w))

        elif isinstance(self.Eg, list):
            self.Eg = np.array(self.Eg)

        self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)
    
        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate *  grad_wrt_w / np.sqrt(self.Eg + self.eps)

    def state_dict(self):
        return {'Eg': self.Eg.tolist()}

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999, m=None, v=None):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = m
        self.v = v
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))
        
        elif isinstance(self.m, list) or isinstance(self.v, list):
            self.m = np.array(self.m)
            self.v = np.array(self.v)

        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return w - self.w_updt
    
    def state_dict(self):
        return {'m': self.m.tolist(),
                'v': self.v.tolist()
            }
