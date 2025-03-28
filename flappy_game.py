import pygame 
import neat
import time
import os
import random
pygame.font.init()
WIN_WIDTH=500

WIN_HEIGHT=800

BIRD_IMGS=[pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
PIPE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG=pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))


STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

class Bird:
    IMGS=BIRD_IMGS
    MAX_ROT=25
    ROT_VEL=20
    ANIMATION_TIME=5
    
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.velocity=0
        self.height=self.y
        self.img_count=0
        self.img=self.IMGS[0]
    def jump(self):
        self.velocity=-10.5
        self.tick_count=0
        self.height=self.y
    def move(self):
        self.tick_count+=1
        d=self.velocity*self.tick_count+1.5*self.tick_count**2
        d=min(d,16)
        
        if d<0:
            d-=2
        self.y=self.y+d
        if d<0 or self.y<self.height+50:
            if self.tilt<self.MAX_ROT:
                self.tilt=self.MAX_ROT
        else:
            if self.tilt>-90:
                self.tilt-=self.ROT_VEL  
    def draw(self,win):
        self.img_count+=1
        if self.img_count<self.ANIMATION_TIME:
            self.img=self.IMGS[0]
        elif self.img_count<2*self.ANIMATION_TIME:
            self.img=self.IMGS[1]
        elif self.img_count<3*self.ANIMATION_TIME:
            self.img=self.IMGS[2]
        elif self.img_count<4*self.ANIMATION_TIME:
            self.img=self.IMGS[1]
        else:
            self.img=self.IMGS[0]
            self.img_count=0
        if self.tilt<=-80:
            self.img=self.IMGS[1]      
            self.img_count=self.ANIMATION_TIME*2
            
        rot_image=pygame.transform.rotate(self.img,self.tilt)
        new_rect=rot_image.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
        
        win.blit(rot_image,new_rect.topleft)
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP=200
    VEL=5
    
    def __init__(self, x):
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
      
        self.passed=False
        self.set_height()
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    
    def move(self):
        self.x-=self.VEL
    
    def draw(self,win):
        win.blit(self.PIPE_TOP,(self.x,self.top))
        win.blit(self.PIPE_BOTTOM,(self.x,self.bottom))
        
    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)

        if b_point or t_point:
            return True

        return False
        
class Base:
    VEL=5
    WIDTH=BASE_IMG.get_width()
    IMG=BASE_IMG
    
    def __init__(self,y):
        self.y=y
        self.x1=0
        self.x2=self.WIDTH
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
            
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

def draw_window(win,bird,pipes,base,score):
    win.blit(BG_IMG,(0,0))
    for pipe in pipes:
        pipe.draw(win)
        
    text = STAT_FONT.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10-text.get_width(), 20))  
    
    base.draw(win)
    
    bird.draw(win)
    pygame.display.update()
      
bird=Bird(200,200)
base=Base(730)
pipes=[Pipe(600)]
win=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
clock = pygame.time.Clock()

run = True
score=0
while run:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird.jump()
    bird.move()
    rem=[]
    add_pipe=False
    draw_window(win, bird, pipes, base,score)
    base.move()
    for pipe in pipes:
        if pipe.x+pipe.PIPE_TOP.get_width()<0:
            rem.append(pipe)
        if not pipe.passed and pipe.x<bird.x:
            pipe.passed=True
            add_pipe=True
            score+=1
        pipe.move()
    if add_pipe:
        pipes.append(Pipe(600))
    for r in rem:
        pipes.remove(r)
    if bird.y+bird.img.get_height()>=730:
        pass
    for pipe in pipes:
        if (pipe.collide(bird,win)):
            run=False
pygame.time.delay(2000)   
pygame.quit()

quit()