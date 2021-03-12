from domrl.engine.game import Game
from domrl.engine.agent import StdinAgent
from domrl.agents.big_money_agent import BigMoneyAgent
from domrl.agents.provincial_agent import ProvincialAgent
import domrl.engine.cards.base as base
from copy import deepcopy

buy_menu_one = [('Gold', 6, 99), ('Witch', 5, 1), ('Council Room', 5, 5), ('Militia', 4, 1), \
            ('Silver', 3, 1), ('Village', 3, 5), ('Silver', 3, 99)]

buy_menu_two = [('Gold', 6, 99), ('Mine', 5, 1), ('Silver', 3, 2), ('Library', 5, 1), \
            ('Village', 3, 1), ('Mine', 5, 1), ('Village', 3, 1), ('Library', 5, 1), \
            ('Village', 3, 2), ('Silver', 3, 99)]

buy_menu_big_money = [('Gold', 6, 99), ('Silver', 3, 99)]

card_set = ["Library", "Mine", "Village", "Militia", "Council Room", "Witch", "Cellar", \
            "Gardens", "Artisan", "Sentry"]

if __name__ == '__main__':
    """
    Run instances of the game.
    """
    num_games = 100
    provincial_one_win_count = 0
    provincial_two_win_count = 0
    num_ties = 0
    for _ in range(num_games):
        kingdoms = [{card:deepcopy(supply_pile) for card,supply_pile in base.BaseKingdom.items() if card in card_set}]

        game = Game(
            agents=[ProvincialAgent(buy_menu = buy_menu_one), ProvincialAgent(buy_menu = buy_menu_two)],
            kingdoms=kingdoms
        )
        state = game.run()
        if state.current_player.total_vp() > state.other_players(state.current_player)[0].total_vp():
            if state.current_player.name == "Player 1":
                provincial_one_win_count += 1
            elif state.current_player.name == "Player 2":
                provincial_two_win_count += 1
        elif state.other_players(state.current_player)[0].total_vp() > state.current_player.total_vp():
            if state.current_player.name == "Player 1":
                provincial_two_win_count += 1
            elif state.current_player.name == "Player 2":
                provincial_one_win_count += 1
        else:
            num_ties += 1

    print("Provincial Buy Menu One won", provincial_one_win_count, "times.")
    print("Provincial Buy Menu Two won", provincial_two_win_count, "times.")
    print("They tied", num_ties, "many times.")

    # print(res)
