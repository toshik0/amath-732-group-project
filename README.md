# A Schooling fish model to Question our Understanding of Invertebrate Diet (S.Q.U.I.D.)
A repository of a simple Python individual-based model to understand how fish schooling can affect squid feeding (no specific species of either).

## How to Run the Code
Using SAF_Model_11.19.ipynb:
1) change the folder in line 9 (os.chdir) to be a folder in your machine
2) adjust any parameters you would like to change (lines 13-53 in the file)

## List of parameters (in order):
### General parameters

**width**: integer; number of grid spaces in the x-direction
**height**: integer; number of grid spaces in the y-direction

### Fish parameters

**num_fish**: integer; number of fish initially
**eat_distance**: float; maximum number of grid spaces a squid needs to be from a fish in order to eat it
**fish_positions**: numpy.array; starting x and y coordinates for every fish
**fish_velocities**:  numpy.array; starting x and y velocities for every fish
**fish_neighbor_dist**: integer; maximum number of grid spaces a fish needs to be from a neighbour to be considered "neighbours" for schooling
**max_fish_speed**: float; maximum speed of the fish in grid spaces per frame

### Fish schooling parameters

**fish_sep_weight**: float; weight for separation behavior (avoiding crowding neighbors/clipping into each other)
**fish_align_weight**: float; weight for alignment behavior (steering towards average directional alignment of neighbors)
**fish_cohesion_weight**: float; weight for cohesion behavior (steering towards average position of neighbors

### Squid parameters (same as fish)

**num_squid**: integer; number of squid initially
**squid_positions**: numpy.array; starting x and y coordinates for every squid
**squid_velocities**:  numpy.array; starting x and y velocities for every squid
**squid_neighbor_dist**: integer; maximum number of grid spaces a squid needs to be from a neighbour to be considered "neighbours" for schooling
**max_squid_speed**: float; maximum speed of the squid in grid spaces per frame

### Squid schooling parameters (same as fish)

**squid_sep_weight**: float; weight for separation behavior (avoiding crowding neighbors/clipping into each other)
**squid_align_weight**: float; weight for alignment behavior (steering towards average directional alignment of neighbors)
**squid_cohesion_weight**: float; weight for cohesion behavior (steering towards average position of neighbors
**close_fish_weight**: float; weight for how much the squid is attracted to the fish

## File Directory

### Main Files

**README.md**
You are here! An overview of all the files in the repository and how to run the code

**SAF_Model_11.19.ipynb**
The most recent model of squid and fish schooling. Start with this one :) The difference between this model and the second-most recent one is this one has squid emojis overlaid with their quivers in the final plot. 

**squid.png**
Squid emoji image used by files when rendering gifs


### ./old_models
Collection of Jupyter Notebooks showcasing incremental progress of model, as well as final python file with commented chunks removed.

**squid_and_fish_model.py**
The base model which is only a Boid model for squid and fish (and the squid can eat the fish).

**SAF_Model_11.12.ipynb**
Adds the "attraction" factor for the squid, so the squid can chase after more lonely or isolated fish.

**SAF_Model_11.17.ipynb**
Fine-tuned the "attraction" factor so that if there is no isolated fish nearby, the squid will go after the nearest school.


### ./animations
Contains output of simulations--several .gif files. Many of these are used in the presentation to showcase changes in model parameters.

**squid_and_fish.gif**
"Baseline" animation featuring slow squid.

**temp.gif**
Temporary file used for testing parameter combinations.

**all the other .gif files**
Sample model outputs showcasing different combinations of parameters.


### ./additional_documentation
LaTeX documents for the "school project" part of the project, as well as the background research notes.

**main.tex**
Shared LaTeX file used to store information about the group project.

**research_notes.tex**
LaTeX file containing all of the notes on different papers read to understand the mechanics of our model, both other models created and observations in the wild.

**writeup.tex**
Shared LaTeX file acting as a summary of the work split and additional information on our model not covered in the presentation.

**references.bib**
BibTeX bibliography document to facilitate the bibliography page in the "research_notes.tex" document.

