import numpy as np

class NoteDetector:
    # reference: https://github.com/yash120394/Computer-Vision/tree/master/Optical%20Music%20Recognition
    
    def __init__(self, spacing_paramter, starting_positions):
        self.spacing_parameter = spacing_paramter
        self.starting_positions = starting_positions
        self.first_spacing_factor_treble = {
            'F': 0,
            'D': 1,
            'B': 2,
            'G': 3,
            'E': 4,
            'A': 2.5,
            'C': 1.5
        }
        self.first_spacing_factor_bass = {
            'F': 1,
            'D': 2,
            'B': 3,
            'G': 4,
            'E': 1.5,
            'A': 0,
            'C': 2.5
        }
        self.treble_clef = self.starting_positions[::2]
        self.bass_clef = self.starting_positions[1::2]
            
    def get_clef(self, detected):
        clefs = {}
        height, width = detected.shape
        for i in range(height):
            for j in range(width):
                if(detected[i,j] == 255):
                    for clef in self.starting_positions:
                        if clef % 2 == 0:
                            if(clef - 3 * self.spacing_parameter < i and i < clef + 8 * self.spacing_parameter):
                                clefs[(i,j)] = 0
                        else:
                            clefs.setdefault((i, j), 1)
        
        return clefs

    def is_note(self, i, clef, max_error, note, is_treble=True):
        if is_treble:
            first_spacing_factor = self.first_spacing_factor_treble[note]
        else:
            first_spacing_factor = self.first_spacing_factor_bass[note]
        for f in [0, 3.5, 3.5]:
            candidate_position_1 = [int(x + first_spacing_factor * self.spacing_parameter + f * self.spacing_parameter - max_error) for x in clef]
            candidate_position_2 = [int(x + first_spacing_factor * self.spacing_parameter + f * self.spacing_parameter - max_error) for x in clef]
            
            for x in range(len(candidate_position_1)):
                if candidate_position_1[x] < i and i < candidate_position_2[x]:
                    return True  

        return False

    def get_notes(self, clef_dict):
        max_error = self.spacing_parameter / 2
        notes = []

        for index, clef in clef_dict.items():
            if(clef==0):
                i, j = index
                
                for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                    if self.is_note(i, self.treble_clef, max_error, note, is_treble=True):
                        notes.append((i, j), note)
                else:
                    l = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                    notes.append((i,j,np.random.choice(l)))
                    
            if(clef==1):
                i, j = index[0], index[1]
                
                for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                    if self.is_note(i, self.bass_clef, max_error, note, is_treble=False):
                        notes.append((i, j), note)
                else:
                    l = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                    notes.append((i,j,np.random.choice(l)))
                
        return notes            
            
    def detect_notes(self, detected):
        clef_dict = self.get_clef(detected)    
        notes = self.get_notes(clef_dict)
        return notes