# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
USAGE:

cd Documents/GitHub/NavigationByReinforcementLearning/Release/

from mapmanager import MapManager
mm = MapManager(height = 84, width = 84 ,firstmap = 2, lastmap = 400, render=False) #to load target images

mm.get_target_image(mapname) # to retrieve the targetimage of the specified map as numpy array, use without argument or with "map00" to retrieve black image

mm.get_random_map() #get a random map in specified range
"""

# ==============================================================================


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import time, random
from abc import ABCMeta, abstractmethod
from vizdoom import *
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imsave

import cv2
import os


from tensorforce.environments import Environment
import tensorforce.util
from PIL import Image

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

pathtomap=os.path.join(__location__, 'gym_vizdoom/envs/D3_exploration_train.wad_manymaps.wad_snapshot.wad') #source of target images

DIRECTORY = os.path.join(__location__, 'targetimages/') #name of directory where targetimages are
MAPSTRING = "map"


class MapManager(Environment):
    """
    Base environment class.
    """
    def __init__(self, height = 84, width = 84 ,firstmap = 2, lastmap = 400, render = False):

        self.iterator = firstmap #increases to iterate through maps start with firstmap
        self.firstmap = firstmap #should be larger or equal 1 , 0 is black image
        self.lastmap = lastmap #last map to generate target from
        self.target_images = None
        self.height = height
        self.width = width
        self.mapstring = MAPSTRING
        self.directory = DIRECTORY
        self.filename = "targetimages_map_" +str(self.firstmap)+"_to_"+ str(self.lastmap)+"_h"+str(self.height)+"w"+str(self.width)+".npy"
        start_time = time.time()
        #check if targets with given specs already exist and load
        try:
            #if exist load from file
            self.target_images = np.load(self.directory+self.filename)
            print ("succesfully loaded file: "+ self.filename + " from " + self.directory)
        except:
            print ("No file found creating new target image file, please stand by...")
            #else load environment and execute target_generator()
            self.render = render #render
            self.env = DoomGame()
            self.env.set_doom_scenario_path(pathtomap)
            self.env.set_doom_map("map02")#MazeMap
            self.env.set_screen_resolution(ScreenResolution.RES_640X480) # 160X120
            self.env.set_screen_format(ScreenFormat.GRAY8)
            self.env.set_render_hud(False)
            self.env.set_render_crosshair(False)
            self.env.set_render_weapon(False)
            self.env.set_render_decals(render)
            self.env.set_render_particles(render)
            self.env.add_available_button(Button.MOVE_FORWARD)
            #self.env.add_available_game_variable(GameVariable.POSITION_Z)
            self.env.add_available_game_variable(GameVariable.AMMO2)
            self.env.add_available_game_variable(GameVariable.POSITION_X)
            self.env.add_available_game_variable(GameVariable.POSITION_Y)
            self.env.set_episode_timeout(100010)
            self.env.set_episode_start_time(10)
            self.env.set_window_visible(render)
            self.env.set_sound_enabled(False)
            self.env.set_living_reward(0)
            self.env.set_mode(Mode.PLAYER)
            self.env.init()
            self.target_generator()
            self.target_saver()
            self.close()
            try:
                # if exist load from file
                self.target_images = np.load(self.directory + self.filename)
            except:
                print("Writing and Loading Failed debug!!")
        start_time = time.time()-start_time
        print("Needed " +  str(start_time)+ " seconds to load/generate target images")
 


    def close(self):
        self.env=None

    def target_saver(self):
        print("Saving new array of size:")
        print(self.target_images.shape)
        processedtis = (self.target_images*255.9).astype('int')
        np.save(self.directory+self.filename, processedtis)

    def initialize_target_array(self):
        """
        #old code not workin!!!
        #shold be a black image instead of image
        s = self.env.get_state().screen_buffer
        s = self.process_image(s)
        #self.target_images = np.stack(s, axis=2)
        self.target_images = np.stack((s,s), axis = 2)
        s_expanded = np.expand_dims(s, axis=2)
        for i in range(2, self.firstmap): #two because of stacking statement
            self.target_images = np.append(self.target_images[:, :, 1:], s_expanded, axis=2)
        """
        self.target_images = np.ones((self.height, self.width, self.lastmap+1))

    def target_generator(self):
        self.initialize_target_array()
        while(self.iterator < self.lastmap+1): #iterate throug all maps
            mapname = self.mapstring +str(self.iterator)
            if (self.iterator <10): #add 0 if below 10
                mapname = self.mapstring+ "0"+ str(self.iterator)
            self.iterator = self.iterator +1
            self.env.set_doom_map(mapname)
            s = self.env.get_state().screen_buffer
            s = self.process_image(s)
            s_expanded = np.expand_dims(s, axis=2)
            self.target_images = np.append(self.target_images[:, :, 1:], s_expanded, axis=2)

    def process_image(self, image):
        s = resize(image, (self.height, self.width))
        return s

    def get_target_image(self, map = "map00"):
        mapnumber = map[3:] #very specific to map name format "map000"
        mapnumber = int(mapnumber)
        temp=self.target_images[:,:, mapnumber]
        ti= temp[:, :, np.newaxis]
        return ti
    '''
    def imagedisplay(self, image, name = "image"):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass
    '''
    def main():
        print("Not supposed to be used as main file")

    def get_random_map(self):

        mapnumber = np.random.randint(self.firstmap, high=self.lastmap + 1)
        if (mapnumber < 10):
            return self.mapstring + "0" + str(mapnumber)
        else:
            return self.mapstring + str(mapnumber)

    if __name__ == '__main__':
        main()

