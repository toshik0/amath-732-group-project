import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Change directory as needed:
#os.chdir("C:\\Users\\mmcclern\\Desktop\\AMATH 732")
os.chdir("C:/Users/rehbe/UW_Masters/Fall_2026/amath-732-group-project")
os.makedirs("animations", exist_ok=True)

## Gif Parameters 
gif_save_path = "many_squid_many_loose_fish.gif"
n_frames = 300 # at 20fps currently

## positions initialized randomly; set seed for reproducibility
np.random.seed(777)

## Box Parameters
width = 100
height = 100


## Fish parameters
num_fish = 200
eat_distance = 1.0 #this is how close the squid need to be to the fish to eat them (should be the same between fish and squid)

fish_positions = np.random.rand(num_fish, 2) * np.array([width, height]) #give the fish a random starting point
fish_velocities = (np.random.rand(num_fish, 2))* np.array([width, height]) #and give them a random starting velocity

fish_neighbor_dist = 10 #this is how close the fish need to be to be considered "neighbors" for schooling
max_fish_speed = .5 #max speed of the fish (needs to be capped or they might go too fast)

## fish schooling params
'''
This is how much each behavior contributes to the overall acceleration of the fish.
This is based on the Boid (bird-oid) model of schooling, which is a basic but decent model of schooling.

- fish_sep_weight: Weight for separation behavior (avoiding crowding neighbors/clipping into each other)
- fish_align_weight: Weight for alignment behavior (steering towards average directional alignment of neighbors)
- fish_cohesion_weight: Weight for cohesion behavior (steering towards average position of neighbors
'''

fish_sep_weight = 0.25
fish_align_weight = 0.5
fish_cohesion_weight = 0.1

## Squid parameters (same as fish)
num_squid = 10
squid_positions = np.random.rand(num_squid, 2) * np.array([width, height])
squid_velocities = (np.random.rand(num_squid, 2))* np.array([width, height])
squid_neighbor_dist = 1.0
max_squid_speed = max_fish_speed*1.1

## squid schooling params (same as fish)
squid_sep_weight = 1
squid_align_weight = 0.5
squid_cohesion_weight = 1.5
close_fish_weight = 2.0 #weight for how much the squid is attracted to the fish (for eating)

## squid emoji
image = plt.imread('squid.png')

# OffsetBox
image_box = OffsetImage(image, zoom=0.025)

def update_fish_movement(positions, velocities, fish_neighbor_dist=5.0, max_speed=4.0, width=10, height=10, eat_distance=1.0,
                         separation_weight=0.05, alignment_weight=1.0, cohesion_weight=1.0,):
    '''
    This is how the fish move. Needs:
    positions: the current positions of the fish
    velocities: the current velocities of the fish
    fish_neighbor_dist: how close the fish need to be to be considered "neighbors" for schooling
    max_speed: the maximum allowed speed of the fish 
    width: the width of the area the fish are in
    height: the height of the area the fish are in
    eat_distance: how close the squid need to be to the fish to eat them
    '''                                          


    #first let the squid try to eat the fish
    for squid_pos in squid_positions:
        fishsquid_distances = np.linalg.norm(positions - squid_pos, axis=1) #dist btwn fish and squid
        close_fish_indices = np.where(fishsquid_distances < eat_distance)[0] #if the fish is close enough, eat it
        
        if len(close_fish_indices) > 0: #then remove the fish from the position and velocity arrays
            positions = np.delete(positions, close_fish_indices, axis=0)
            velocities = np.delete(velocities, close_fish_indices, axis=0)

    separation = np.zeros_like(velocities)
    alignment = np.zeros_like(velocities)
    cohesion = np.zeros_like(velocities)
            
    for i in range(len(positions)):
      #THEN update fish movement based on the remaining fish
        dist = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
        neighbor_mask = (dist > 0) & (dist < fish_neighbor_dist) #distance between fish and their neighbors
                       
        idxs = np.where(neighbor_mask[i])[0]
        if len(idxs) == 0:
            continue

        dvec = positions[i] - positions[idxs]
        dists = dist[i, idxs][:, np.newaxis]
        separation[i] = np.sum(dvec / (dists**2 + 1e-5), axis=0) #stop the fish from clipping into each other
        if len(idxs) > 0:
            alignment[i] = np.nanmean(velocities[idxs], axis=0) - velocities[i]
            cohesion[i] = np.nanmean(positions[idxs], axis=0) - positions[i]
   
        else:
            alignment[i] = np.zeros_like(velocities[i])
            cohesion[i] = np.zeros_like(positions[i])

    accel = (fish_sep_weight * separation + 
             fish_align_weight * alignment + 
             fish_cohesion_weight * cohesion) #acceleration with weights

    velocities = velocities + accel #update velocities based on acceleration
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = velocities / np.maximum(speeds, 1e-5) * np.minimum(speeds, max_speed) #cap the speed of the fish so they don't go too fast or too slow

    positions = (positions + velocities) % np.array([width, height]) #update position

    # ^^^ will need to change BCs to make more tank like!! Or can keep as periodic for now

    return positions, velocities,  #return the position and velocity of the fish


def update_squid_movement(positions, velocities, neighbor_dist=5.0, eat_distance=2.0, max_speed=4.0,
    width=10, height=10, separation_weight=0.05, alignment_weight=1.0, cohesion_weight=1.0):
    '''
    This is how the squids move. Needs:
    positions: the current positions of the squids
    velocities: the current velocities of the squids
    neighbor_dist: how close the squids need to be to be considered "neighbors" for schooling
    eat_distance: the distance at which squids will see the fish to eat them
    max_speed: the maximum allowed speed of the squids

    width: the width of the area the squids are in *
    height: the height of the area the squids are in *
    (don't need eat_distance for squid since their pop'n isn't dependent on getting eaten)
    * need to be the same as the fish or it will break the sim
    '''
    
    separation = np.zeros_like(velocities)
    alignment = np.zeros_like(velocities)
    cohesion = np.zeros_like(velocities)
    attraction = np.zeros_like(velocities) # Steer each squid toward nearby fish.


    for i in range(len(positions)):
        
        fish_dist = np.linalg.norm(fish_positions - positions[i], axis=1)
        nearby = fish_dist < eat_distance
    
        if np.any(nearby):
            nearby_fish = fish_positions[nearby]
            if nearby_fish.size > 0:
                # Prefer fish that are less crowded when deciding where to chase.
                crowded_mask = (np.linalg.norm(nearby_fish[:, None, :] - fish_positions[None, :, :], axis=-1) < fish_neighbor_dist).any(axis=1)
                lonely_mask = ~crowded_mask

            if np.any(lonely_mask):
                lonely_target = np.nanmean(nearby_fish[lonely_mask], axis=0)
                crowded_target = (
                np.nanmean(nearby_fish[crowded_mask], axis=0)
                if np.any(crowded_mask)
                else lonely_target
                )
                attraction[i] = (0.9 * lonely_target + 0.1 * crowded_target) - positions[i]
            else:
                attraction[i] = np.nanmean(nearby_fish, axis=0) - positions[i]

    
        dist = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)

        idxs = np.where((dist[i] > 0) & (dist[i] < neighbor_dist))[0]
        dists = dist[i, idxs, np.newaxis]
        separation[i] = np.sum((positions[i] - positions[idxs]) / (dists**2 + 1e-5), axis=0)
        if len(idxs) > 0:
            alignment[i] = np.nanmean(velocities[idxs], axis=0) - velocities[i]
            cohesion[i] = np.nanmean(positions[idxs], axis=0) - positions[i]
        else:
            alignment[i] = np.zeros_like(velocities[i])
            cohesion[i] = np.zeros_like(positions[i])


    accel = (squid_sep_weight * separation + 
             squid_align_weight * alignment + 
             squid_cohesion_weight * cohesion + 
             close_fish_weight * attraction)

    velocities = velocities + accel
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = velocities / np.maximum(speeds, 1e-5) * np.minimum(speeds, max_speed)
    positions = (positions + velocities) % np.array([width, height])

    return positions, velocities


fig, ax = plt.subplots(figsize=(7, 7))

def animate(_frame):
    global fish_positions, fish_velocities, fish_quiver

    ax.clear()
    ax.set(xlim=(0, width), ylim=(0, height), 
        title=f"Squid and Fish Schooling (Fish: {len(fish_positions)}, Squid: {num_squid})")
    ax.set_aspect("equal")

    fish_positions, fish_velocities = update_fish_movement(
        fish_positions,
        fish_velocities,
        fish_neighbor_dist=fish_neighbor_dist,
        max_speed=max_fish_speed,
        width=width,
        height=height,
        eat_distance=1.0,
        separation_weight=fish_sep_weight,
        alignment_weight=fish_align_weight,
        cohesion_weight=fish_cohesion_weight)

    fish_quiver = ax.quiver(
        fish_positions[:, 0], fish_positions[:, 1],
        fish_velocities[:, 0], fish_velocities[:, 1],
        angles="xy", scale_units="width", scale=50,)


    global squid_positions, squid_velocities, squid_quiver

    squid_positions, squid_velocities = update_squid_movement(
        squid_positions,
        squid_velocities,
        neighbor_dist=squid_neighbor_dist,
        max_speed=max_squid_speed,
        width=width,
        height=height,
        eat_distance=eat_distance,
        separation_weight=squid_sep_weight,
        alignment_weight=squid_align_weight,
        cohesion_weight=squid_cohesion_weight)

    #squid_quiver.remove()
    squid_quiver = ax.quiver(
        squid_positions[:, 0], squid_positions[:, 1],
        squid_velocities[:, 0], squid_velocities[:, 1],
        angles="xy", scale_units="width", scale=50, color="blue")

    for x0, y0 in zip(squid_positions[:, 0], squid_positions[:, 1]):
        squid_image = AnnotationBbox(image_box, (x0,y0), frameon=False)
        ax.add_artist(squid_image)

fish_anim = FuncAnimation(fig, animate, frames=n_frames, interval=50)

fish_anim.save(os.path.join("animations", gif_save_path), writer="pillow", fps=20)

plt.show()

