"""两个人工智能黑白棋互相博弈，生成数据集第三问需要的"""
import sys
from pygame.locals import *
import random
import pygame.gfxdraw
from AI_alpha_beta import *
from Checkboard import *
from DrawUI import *
import numpy as np
import pandas as pd
import csv

Chessman = namedtuple('Chessman', 'Name Value Color')
Point = namedtuple('Point', 'X Y')
BLACK_CHESSMAN = Chessman('黑子', 1, (45, 45, 45))
WHITE_CHESSMAN = Chessman('白子', 2, (255, 255, 255))

offset = [(1, 0), (0, 1), (1, 1), (1, -1)]

def print_text(screen, font, x, y, text, fcolor=(255, 255, 255)):
    imgText = font.render(text, True, fcolor)
    screen.blit(imgText, (x, y))

CURRENT_BLACK_POINT = Point(-1,-1)
CURRENT_WHITE_POINT = Point(-1,-1)




def save_dataset(score):

    f1 = pd.read_csv('x_train.csv', engine='python')
    f2 = pd.read_csv('xxxx.csv')
    f3= pd.read_csv('y_train.csv')
    """file = [f1, f2]
    train = pd.concat(file)
    train.to_csv('x_train.csv', index=0, sep=',')"""
    with open('x_train.csv', 'ab') as f:
        f.write(open('xxxx.csv', 'rb').read())  # 将2.csv内容追加到1.csv的后面
    if score==0:
        yyyy=[0]
    else:
        yyyy=[1]
    yyyy = np.array(yyyy, dtype=np.ubyte)
    yyyy = yyyy.reshape((1, -1))
    file = 'yyyy.csv'
    np.savetxt(file, yyyy, fmt='%d', delimiter=',')
    with open('y_train.csv', 'ab') as f:
        f.write(open('yyyy.csv', 'rb').read())  # 将2.csv内容追加到1.csv的后面



def main():
    pygame.init()

    # 文件x_trian, y_train
    fx = 'x_train.txt'
    fy = 'y_train.txt'

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

    computer_white = ChessAI(Line_Points, WHITE_CHESSMAN)
    computer_black = ChessAI(Line_Points, BLACK_CHESSMAN)

    step = 1
    cccc = 0
    #让AI对局n次
    GAME_MAX_COUNT = 300
    while True :
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
            white_point, score_white = computer_white.findBestChess(WHITE_CHESSMAN.Value)  # 2

            CURRENT_WHITE_POINT = white_point
            # 判断是否赢得比赛
            winner = checkerboard.drop(cur_runner, white_point)
            if white_point == (-1, -1):
                winner = _get_next(cur_runner)
            if winner is not None :
                print("白棋赢了！")
                save_dataset(1)
                cccc=cccc+1
                white_win_count += 1
            cur_runner = _get_next(cur_runner)

            computer_black.get_opponent_drop(CURRENT_WHITE_POINT)
            if winner is None :
                black_point, score_black = computer_black.findBestChess(BLACK_CHESSMAN.Value)  # 1

                # 判断是否赢得比赛
                CURRENT_BLACK_POINT = black_point
                winner = checkerboard.drop(cur_runner, black_point)
                if black_point == (-1, -1):
                    winner = _get_next(cur_runner)
                if winner is not None:
                    print("黑棋赢了！")
                    save_dataset(0)
                    cccc=cccc+1
                    black_win_count += 1

                cur_runner = _get_next(cur_runner)

        # 有赢家，初始化下一局
        if winner is not None:
            if black_win_count + white_win_count == 15:
                exit(0)
            winner = None
            cur_runner = BLACK_CHESSMAN
            checkerboard = Checkerboard(Line_Points)
            step = 1
            computer_white = ChessAI(Line_Points, WHITE_CHESSMAN)
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
