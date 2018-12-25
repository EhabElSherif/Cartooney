#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:28:45 2018

@author: Haneen
"""
#import os
import kivy
kivy.require('1.10.1')
#kivy imports
from kivy.app import App
from kivy.lang import Builder
#from kivy.factory import Factory
from kivy.uix.popup import Popup
#from kivy.uix.button import Button
from kivy.core.text import LabelBase
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
#from kivy.uix.filechooser import FileChooserListView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)

from kivy.config import Config
Config.set('graphics', 'width', '1400')
Config.set('graphics', 'height', '800')

LabelBase.register(name = "Bigfish", fn_regular = "Bigfish.ttf")

#import os
img = 'accent.png'

class CustPopup(FloatLayout):
    cancel = ObjectProperty(None)
    choose = ObjectProperty(None)
    
class ScreenManage(ScreenManager):
    pass

class WelcomeScreen(Screen):
    pass

class MainScreen(Screen):
    
    __popup = Popup()
    def loadButton(self):
#        self.__file_chooser = FileChooserListView(rootpath = '/Users/Haneen/') #change directory******************
#        self.__file_chooser.on_submit = self.submitted()
        self.__custpop = CustPopup(cancel = self.dismiss, choose = self.choose)
        self.__custpop.ids.cancel = self.dismiss()
#        self.__custpop.ids.choose.bind(on_press = self.choose())
        self.__popup = Popup(title = "load file", content = self.__custpop, size_hint=(0.9, 0.9))
        self.__popup.open()
        
    def dismiss(self):
        self.__popup.dismiss()
    
    def choose(self):
        self.__popup.dismiss()
        self.ids.img.source = (self.__custpop.ids.filechooser.selection)[0]
        img = (self.__custpop.ids.filechooser.selection)[0]
        
class CartoonScreen(Screen):
    pass

class MaskScreen(Screen):
    pass

class EffectScreen(Screen):
    pass

screen_manager = Builder.load_file("cartooney.kv")

class CartooneyApp(App):
    def build(self):
        return screen_manager


app = CartooneyApp()
app.run()