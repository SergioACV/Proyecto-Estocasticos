class Simulator:
    """Simulación dinámica con pasos discretos."""
    
    def __init__(self, model, initial_state, dt=1.0, total_time=168.0):
        self.model = model
        
        # Estado poblacional (no por nodo)
        self.S, self.I = initial_state
        
        self.dt = dt
        self.total_time = total_time
        self.time = 0
        
        self.history = []  # guardamos evolución temporal

    def step(self):
        """Avanza la simulación un paso en el tiempo."""
        
        dS = self.model.dS_dt(self.S, self.I)
        dI = self.model.dI_dt(self.S, self.I)
        
        # Actualizamos estado con método de Euler
        self.S += dS * self.dt
        self.I += dI * self.dt
        

        self.time += self.dt
        self.history.append((self.time, self.S, self.I))

    def run(self):
        """Corre la simulación completa."""
        while self.time < self.total_time:
            self.step()
