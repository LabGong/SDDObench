class Config:
    def __init__(self, **kwargs):
        # peak height and width
        self.lb_h = kwargs.get("lb_h", -70)
        self.ub_h = kwargs.get("ub_h", -30)
        self.lb_w = kwargs.get("lb_w", 1)
        self.ub_w = kwargs.get("ub_w", 12)

        # dynamic control
        self.t_interval = kwargs.get("t_interval", 1)
        self.t_origin = kwargs.get("t_origin", 0)

        # peak number 
        self.max_dim = kwargs.get("max_dim", 50)
        self.lb_num_peaks = kwargs.get("lb_num_peaks", 3)
        self.ub_num_peaks = kwargs.get("ub_num_peaks", 40)

        # transformation params
        self.change_rate = kwargs.get("change_rate", 0.3)
        self.severity_h = kwargs.get("severity_h", 0.3)
        self.severity_w = kwargs.get("severity_w", 0.1)
        self.severity_p = kwargs.get("severity_p", 0.5)
        self.severity_x = kwargs.get("severity_x", 0.4)

        # drift params
        self.epsilon = kwargs.get("epsilon", 0.05)  
        self.k = kwargs.get("k", 5)  


