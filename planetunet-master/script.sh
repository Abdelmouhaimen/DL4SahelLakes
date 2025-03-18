#!/bin/bash
cd /home/mathilde/Bureau/asarhane/planetunet-master


eval 'python main_custom.py --config config.custom.config_landsat5_64'

eval 'python main_custom.py --config config.custom.config_landsat5_128_32_70'

eval 'python main_custom.py --config config.custom.config_landsat5_128_32_60'

eval 'python main_custom.py --config config.custom.config_landsat5_256_16_60'


echo "Command sequence finished succesfully"
