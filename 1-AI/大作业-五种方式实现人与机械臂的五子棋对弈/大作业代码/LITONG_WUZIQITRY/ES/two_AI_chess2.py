"""两个人工智能黑白棋互相博弈，生成数据集第三问需要的"""
import sys
from pygame.locals import *
import random
import pygame.gfxdraw
from AI_alpha_beta  import *
from White_AI_alpha_beta  import *
from Checkboard import *
from DrawUI import *
import numpy as np
import pandas as pd
import csv
import time


Chessman = namedtuple('Chessman', 'Name Value Color')
Point = namedtuple('Point', 'X Y')
BLACK_CHESSMAN = Chessman('黑子', 1, (45, 45, 45))
WHITE_CHESSMAN = Chessman('白子', 2, (255, 255, 255))

offset = [(1, 0), (0, 1), (1, 1), (1, -1)]
wwin={0:0,1:0,2:0,3:0,4:0}
def print_text(screen, font, x, y, text, fcolor=(255, 255, 255)):
    imgText = font.render(text, True, fcolor)
    screen.blit(imgText, (x, y))

CURRENT_BLACK_POINT = Point(-1,-1)
CURRENT_WHITE_POINT = Point(-1,-1)



def main():
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('五子棋')

    font1 = pygame.font.SysFont('SimHei', 32)
    font2 = pygame.font.SysFont('SimHei', 72)
    fwidth, fheight = font2.size('黑方获胜')

    checkerboard = Checkerboard(Line_Points)
    cur_runner = BLACK_CHESSMAN
    winner = None

    black_win_count = 0
    white_win_count = 0
    X_TRAIN = np.zeros((1, 361), dtype=int)

    computer_white = WChessAI(Line_Points, WHITE_CHESSMAN)
    #computer_black = BChessAI(Line_Points, BLACK_CHESSMAN)
    computer_black = ChessAI(Line_Points, BLACK_CHESSMAN)

    step = 1
    cccc = 0
    count = 0
    #让AI对局n次
    GAME_MAX_COUNT =10

    while True :
        global bwin
        global wwin
        #选择对战的
        w1='E:/python_code/LITONG_WUZIQITRY/ES/w1.txt'
        w2='E:/python_code/LITONG_WUZIQITRY/ES/w2.txt'
        w3 = 'E:/python_code/LITONG_WUZIQITRY/ES/w3.txt'
        w4 = 'E:/python_code/LITONG_WUZIQITRY/ES/w4.txt'
        w5 = 'E:/python_code/LITONG_WUZIQITRY/ES/w5.txt'

        W=[w1,w2,w3,w4,w5]
        if count==0:
            wfilename =w1

        elif count==1:
            wfilename = w1

        elif count==2:
            wfilename = w1

        elif count==3:
            wfilename = w1

        elif count==4:
            wfilename = w1

        elif count==5:
            wfilename = w2

        elif count==6:
            wfilename = w2

        elif count==7:
            wfilename = w2

        elif count==8:
            wfilename = w2

        elif count==9:
            wfilename = w2

        elif count==10:
            wfilename = w3

        elif count==11:
            wfilename = w3

        elif count==12:
            wfilename = w3

        elif count==13:
            wfilename = w3

        elif count==14:
            wfilename = w3

        elif count==15:
            wfilename = w4

        elif count==16:
            wfilename = w4

        elif count==17:
            wfilename = w4

        elif count==18:
            wfilename = w4

        elif count==19:
            wfilename = w4

        elif count==20:
            wfilename = w5

        elif count==21:
            wfilename = w5

        elif count==22:
            wfilename = w5

        elif count==23:
            wfilename = w5

        elif count==24:
            wfilename = w5


        if cccc > GAME_MAX_COUNT :
            print(cccc)
            break
        if step == 1:
            x = random.randint(0, 18)
            y = random.randint(0, 18)
            black_point = Point(x, y)
            computer_black.board[x][y] = WHITE_CHESSMAN.Value
            winner = checkerboard.drop(cur_runner, black_point)
            CURRENT_BLACK_POINT = black_point
            cur_runner = _get_next(cur_runner)
            step = 0

        if winner is None:
            computer_white.get_opponent_drop(CURRENT_BLACK_POINT)
            white_point, score_white = computer_white.findBestChess(WHITE_CHESSMAN.Value,wfilename)  # 2

            CURRENT_WHITE_POINT = white_point
            # 判断是否赢得比赛
            winner = checkerboard.drop(cur_runner, white_point)
            if white_point == (-1, -1):
                winner = _get_next(cur_runner)
            if winner is not None :
                print("白棋赢了！")
                wwin[count % 5] = wwin[count % 5] + 1
            cur_runner = _get_next(cur_runner)

            computer_black.get_opponent_drop(CURRENT_WHITE_POINT)
            if winner is None :
                #black_point, score_black = computer_black.findBestChess(BLACK_CHESSMAN.Value,bfilename)  # 1
                black_point, score_black = computer_black.findBestChess(BLACK_CHESSMAN.Value)
                # 判断是否赢得比赛
                CURRENT_BLACK_POINT = black_point
                winner = checkerboard.drop(cur_runner, black_point)
                if black_point == (-1, -1):
                    winner = _get_next(cur_runner)
                if winner is not None:
                    print("黑棋赢了！")
                cur_runner = _get_next(cur_runner)

        # 有赢家，初始化下一局
        if winner is not None:
            ###############
            #time.sleep(5)

            #if winner.Name=='黑子':
            #    bwin[count//5]=bwin[count//5]+1
            #else:
            #    wwin[count%5]=wwin[count%5]+1

            count=count+1

            if count>24:

                wwin = sorted(wwin.items(), key=lambda x: x[1], reverse=True)  # 按字典集合中，每一个元组的第二个元素排列。
                # 胜率进行排序
                temp=np.loadtxt(W[ wwin[0][0 ] ], dtype=np.float, delimiter=',', unpack=False)
                temp = temp.reshape((14, 1))
                np.savetxt('w1.txt', temp, fmt='%f', delimiter=',')
                #最好的突变
                for i in range(14):
                    temp2 = random.uniform(-0.5 * temp[i], 0.5 * temp[i])
                    if temp2 < 1:
                        temp2 = 1
                    temp[i] = temp[i] + temp2
                np.savetxt('w4.txt', temp, fmt='%f', delimiter=',')

                temp = np.loadtxt(W[wwin[1][0 ]], dtype=np.float, delimiter=',', unpack=False)
                temp = temp.reshape((14, 1))
                np.savetxt('w2.txt', temp, fmt='%f', delimiter=',')
                # 最好的突变
                for i in range(14):
                    temp2 = random.uniform(-0.5 * temp[i], 0.5 * temp[i])
                    if temp2<1:
                        temp2=1
                    temp[i] = temp[i] + temp2
                np.savetxt('w5.txt', temp, fmt='%f', delimiter=',')

                temp = np.loadtxt(W[wwin[2][0]], dtype=np.float, delimiter=',', unpack=False)
                temp = temp.reshape((14, 1))
                np.savetxt('w3.txt', temp, fmt='%f', delimiter=',')

                count=0
                cccc=cccc+1
                print(wwin)
                wwin = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            #控制循环

            ##############
            """if black_win_count + white_win_count == 15:
                exit(0)"""
            winner = None
            cur_runner = BLACK_CHESSMAN
            checkerboard = Checkerboard(Line_Points)
            step = 1
            computer_white = WChessAI(Line_Points, WHITE_CHESSMAN)
            #computer_black = BChessAI(Line_Points, BLACK_CHESSMAN)
            computer_black = ChessAI(Line_Points, BLACK_CHESSMAN)

        # 画棋盘
        DrawUI._draw_checkerboard(screen)

        # 画棋盘上已有的棋子
        for i, row in enumerate(checkerboard.checkerboard):
            for j, cell in enumerate(row):
                if cell == BLACK_CHESSMAN.Value:
                    DrawUI._draw_chessman(screen, Point(j, i), BLACK_CHESSMAN.Color)
                elif cell == WHITE_CHESSMAN.Value:
                    DrawUI._draw_chessman(screen, Point(j, i), WHITE_CHESSMAN.Color)

        # 在右侧打印出战况
        DrawUI._draw_left_info(screen, font1, cur_runner, black_win_count, white_win_count,DrawUI)

        # 判断最终的冠军
        if winner:
            print_text(screen, font2, (SCREEN_WIDTH - fwidth)//2, (SCREEN_HEIGHT - fheight)//2, winner.Name + '获胜', RED_COLOR)


        pygame.display.flip()


def _get_next(cur_runner):
    if cur_runner == BLACK_CHESSMAN:
        return WHITE_CHESSMAN
    else:
        return BLACK_CHESSMAN

if __name__ == '__main__':
    main()
