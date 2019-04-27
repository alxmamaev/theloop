from prettytable import PrettyTable

def start_theloop():
    print("""   _____ _______       _____ _______   _______ _    _ ______ _      ____   ____  _____
  / ____|__   __|/\   |  __ \__   __| |__   __| |  | |  ____| |    / __ \ / __ \|  __ \\
 | (___    | |  /  \  | |__) | | |       | |  | |__| | |__  | |   | |  | | |  | | |__) |
  \___ \   | | / /\ \ |  _  /  | |       | |  |  __  |  __| | |   | |  | | |  | |  ___/
  ____) |  | |/ ____ \| | \ \  | |       | |  | |  | | |____| |___| |__| | |__| | |
 |_____/   |_/_/    \_\_|  \_\ |_|       |_|  |_|  |_|______|______\____/ \____/|_|\n""")


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


def early_stopping():
    print("EARLY STOPPING")
