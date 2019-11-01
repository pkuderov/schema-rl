from model.schemanet import SchemaNet

from player import Player
if __name__ == '__main__':
    model = SchemaNet()
    reward_model = SchemaNet()
    standard_player = Player(model, reward_model)
    standard_player.play(log=True, cheat=False)
