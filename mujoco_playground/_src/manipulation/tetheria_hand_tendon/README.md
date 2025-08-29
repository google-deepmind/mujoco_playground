# LEAP Hand environments

## Hardware

* [Dynamixel XL330-M288-T](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
* [PD gain table](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#position-pid-gain80-82-84)



## software
### data transformation from motor encoder [0, 65535] to tendon length
Before runnin**data** folder.
```
python data_transform.py
```

### pd tunning
First make sure you have installed everything based on the readme file in the main folder.
Run pd tuning for sim to real gap:
```
cd teteria_hand_tendon
python pd_autotune.py
```


### step response
```
python step_response_tendon.py
```

step response of hand from real motor input:
```
python step_response_tendon_from_real_control.py
```
The aim of this function is to see the gap between the real system and the simulated system.




# Notes
Whenever you want to install a package in the environment, use
```
uv pip install package_name
```
Only `pip` does not work.
