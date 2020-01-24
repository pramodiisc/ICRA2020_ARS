# Source https://nerdparadise.com/programming/pygame/part4


import pygame
import math
import time
import numpy as np

def create_background(width, height):
    colours = [(255,255,255), (212,212,212)]
    background = pygame.Surface((width, height))
    pygame.draw.rect(background,(255, 255, 255), pygame.Rect(0,0,width, height))
    return background


def is_trying_to_quit(event):
    pressed_keys = pygame.key.get_pressed()
    alt_pressed = pressed_keys[pygame.K_LALT] or pressed_keys[pygame.K_RALT]
    x_button = event.type == pygame.QUIT
    altF4 = alt_pressed and event.type == pygame.KEYDOWN and event.key == pygame.K_F4
    escape = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
    return x_button or altF4 or escape


def run_program(width, height, fps):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    background = create_background(width, height)
    clock = pygame.time.Clock()
    x = 200
    y = 150
    theta = 0

    while True:
        for event in pygame.event.get():
            if is_trying_to_quit(event):
                return
        screen.blit(background, (0,0))
        theta += 0.01
        x += 0.1
        draw_quadruped(screen, x, y, theta)
        pygame.display.flip()
        clock.tick(fps)
    return 

# Rotation matrix in plane to convert from body-fixed frame to space-fixed frame
def rotMatrix(theta):
    return np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])



def draw_quadruped(surface, x, y, theta):
    colour = (212, 212, 212)
    p1 = 30*np.array([-1, -1])
    p2 = 30*np.array([-1, 1])
    p3 = 30*np.array([1, 1])
    p4 = 30*np.array([1, -1])

    cen = np.array([x, y])

    R = rotMatrix(theta)
    p1s = np.dot(R, p1)
    p2s = np.dot(R, p2)
    p3s = np.dot(R, p3)
    p4s = np.dot(R, p4)

    points_list = [cen + p1s, cen + p2s, cen + p3s, cen + p4s] 

    pygame.draw.polygon(surface, colour, points_list)
    for i in range(4):
        pygame.draw.circle(surface, (255, 0, 0), points_list[i].astype(int), 10)




run_program(1200, 600, 60)
