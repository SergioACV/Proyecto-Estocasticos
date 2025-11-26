class Node:
    """Representa un nodo individual en la red."""
    def __init__(self, state="S"):
        self.state = state  # "S", "I", "R"

    def update_state(self, new_state):
        # Aquí después pondremos reglas de transición
        self.state = new_state
