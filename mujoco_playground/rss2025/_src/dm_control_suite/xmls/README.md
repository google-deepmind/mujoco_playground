# XML Changes Compared to DM Control

Several changes are made to DM Control envs for them to be performant on XPU, although changes are kept to a minimum. For all envs, we disable `eulerdamp` and reduce solver `iterations` and `ls_iterations`. For a few envs, we increase the simulator timestep, and set `max_contact_points` and `max_geom_pairs` for contact culling. The full set of changes are listed below:

* acrobot
  * `iterations`="2", `ls_iterations`="4`
* ball_in_cup
  * `iterations`="1", `ls_iterations`="4"
* cartpole
  * `iterations`="1", `ls_iterations`="4"
* cheetah
  * `iterations`="4", `ls_iterations`="8"
  * `max_contact_points`=6, `max_geom_pairs`=4
* finger
  * `iterations`="2", `ls_iterations`="8"
  * `max_contact_points`=4, `max_geom_pairs`=2
  * change cylinder contype/conaffinity to 0
* fish
  * `iterations`="2", `ls_iterations`="6"
  * contacts disabled
* hopper
  * `iterations`="4", `ls_iterations`="8"
  * `max_contact_points`=6, `max_geom_pairs`=2
* humanoid
  * Removed touch sensors
  * `timestep`="0.005" compared to 0.0025 in DM Control
  * `max_contact_points`=8, `max_geom_pairs`=8
* manipulator
  * `timestep`="0.005" from "0.002" in DM Control
  * `max_contact_points`=8, `max_geom_pairs`=8
* pendulum
  * `timestep`="0.01" compared to "0.02" in DM Control
  * `iterations`="4", `ls_iterations`="8"
* point_mass
  * `iterations`="1", `ls_iterations`="4"
* reacher
  * `timestep`="0.005" from "0.02" in DM Control
  * `iterations`="1", `ls_iterations`="6"
* swimmer
  * `timestep`="0.003" from "0.002" in DM Control
  * `iterations`="4", `ls_iterations`="8"
  * geom `contype`/`conaffinity` set to zero, since contacts are disabled
* walker
  * `timestep`="0.005", from "0.0025" in DM Control
  * `iterations`="2", `ls_iterations`="5"
  * `max_contact_points`=4, `max_geom_pairs`=4
