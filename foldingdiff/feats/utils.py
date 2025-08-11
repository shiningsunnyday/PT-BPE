

HYDROPATHY = {
    # Kyte–Doolittle index
    "A": 1.8, "C": 2.5, "D":-3.5, "E":-3.5, "F": 2.8, "G":-0.4,
    "H":-3.2, "I": 4.5, "K":-3.9, "L": 3.8, "M": 1.9, "N":-3.5,
    "P":-1.6, "Q":-3.5, "R":-4.5, "S":-0.8, "T":-0.7, "V": 4.2,
    "W":-0.9, "Y":-1.3, "X": 0.0, "-": 0.0,
}


def zero_initialize(self):
    # Loop over all named parameters and zero them
    for name, param in self.named_parameters():
        nn.init.constant_(param, 0.0)

def l1_penalty(self):
    """
    Compute the sum of absolute values of all parameters in this module.
    """
    l1_sum = 0.0
    for param in self.parameters():
        l1_sum += param.abs().sum()
    return l1_sum        

