from domrl.engine.game import Game
from domrl.engine.agent import StdinAgent
from domrl.agents.big_money_agent import BigMoneyAgent
from domrl.agents.provincial_agent import ProvincialAgent
import domrl.engine.cards.base as base

def get_card_set(cards):
    dict = {}
    for key, val in base.BaseKingdom.items():
        if key in cards:
            dict[key] = val
    return dict

card_set = get_card_set(["Witch", "Council Room", "Militia", "Village", "Gardens", "Chapel", \
                        "Artisan", "Sentry", "Bureaucrat", "Cellar"])

buy_menu= [('Gold', 6, 99), ('Witch', 5, 1), ('Council Room', 5, 5), ('Militia', 4, 1), \
            ('Silver', 3, 1), ('Village', 3, 5), ('Silver', 3, 99)]

if __name__ == '__main__':
    """
    Run instances of the game.
    """
    num_games = 100000
    provincial_win_count = 0
    big_money_win_count = 0
    for i in range(0, num_games):
        game = Game(
            agents=[ProvincialAgent(buy_menu = buy_menu), BigMoneyAgent()],
            kingdoms=[card_set],
        )
        _, player = game.run()
        if player.name == "Player 1":
            provincial_win_count += 1
        elif player.name == "Player 2":
            big_money_win_count += 1
        print("game over2")

    print("Provincial won ", provincial_win_count, " many times.")
    print("Big Money won ", big_money_win_count, " many times.")

    # print(res)
