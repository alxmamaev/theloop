from prettytable import PrettyTable

def start_theloop(experiment_name, experiment_id, n_epoch):
    print(f"""   _____ _______       _____ _______   _______ _    _ ______ _      ____   ____  _____
  / ____|__   __|/\   |  __ \__   __| |__   __| |  | |  ____| |    / __ \ / __ \|  __ \\
 | (___    | |  /  \  | |__) | | |       | |  | |__| | |__  | |   | |  | | |  | | |__) |
  \___ \   | | / /\ \ |  _  /  | |       | |  |  __  |  __| | |   | |  | | |  | |  ___/
  ____) |  | |/ ____ \| | \ \  | |       | |  | |  | | |____| |___| |__| | |__| | |
 |_____/   |_/_/    \_\_|  \_\ |_|       |_|  |_|  |_|______|______\____/ \____/|_|


EXPERIMENT NAME: {experiment_name}
EXPERIMENT ID: {experiment_id}
NUM EPOCH: {n_epoch}""")


def rabbit(message):
    print(f"  |￣￣￣￣￣￣|\n    {message}  \n  |＿＿＿＿＿＿|\n(\\__/) || \n(•ㅅ•) || \n/ 　 づ")


def log_metrics(title, metrics):
    ptable = PrettyTable()
    ptable.title = title
    ptable.field_names = ["Metric", "Vlaue"]

    for m, v in metrics.items():
        ptable.add_row([m, float(v)])

    print(ptable)

def delimeter():
    print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n\n")
