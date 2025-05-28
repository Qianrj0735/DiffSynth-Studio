# MineRL Navigate Video Dataset

The MineRL dataset was crowd sourced by Guss et al. (2019) for reinforcement learning applications. The dataset shows human players traveling to goal coordinates in procedurally generated 3D worlds of the video game Minecraft, traversing forests, mountains, villages, and oceans.

To create a video prediction dataset, we combined the human demonstrations for the `Navigate` and `Navigate Extreme` tasks and split them into non-overlapping sequences of length 500. The dataset contains 961 training videos and 225 test videos as individual MP4 files. Additional metadata is stored in JSON format and contains the actions taken by the players in the game and the angle between the forward direction and the direction to the goal.

References:

```
@misc{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```

```
@article{guss2019minerl,
  title={Mine{RL}: A Large-Scale Dataset of {M}inecraft Demonstrations},
  author={William H. Guss and Houghton, Brandon and Topin, Nicholay and Wang, Phillip and Codel, Cayden and Veloso, Manuela and Salakhutdinov, Ruslan},
  journal={International Joint Conference on Artificial Intelligence},
  year={2019},
}
```

