import pygame

pygame.init()
size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Video Test")

WHITE = (255, 255, 255)

windowOpen = True
while windowOpen:
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT:
            windowOpen = False

    screen.fill(WHITE)
    pygame.display.flip()

pygame.quit()