import torch


def calculate_accuracy_for_each_class(num_classes: int, model_l: torch.nn.Module, pred_dl):
    acc_dict = {}
    total_dict = {}

    for i in range(0, num_classes):
        acc_dict[f"{i}"] = 0
        total_dict[f"{i}"] = 0

    final_dict = {}

    with torch.no_grad():

        for batch in pred_dl:

            x, y = batch
            y_hat = model_l(x)
            y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)

            for i in range(len(y_hat)):
                if y_hat[i] == y[i]:
                    acc_dict[f"{y[i]}"] = acc_dict.get(f"{y[i]}", 0) + 1

                total_dict[f"{y[i]}"] = total_dict.get(f"{y[i]}", 0) + 1

        for key, value in acc_dict.items():

            total = total_dict[key]

            final_dict[key] = 0

            if total > 0:
                final_dict[key] = value / total

    return final_dict


def display_accuracy_chart(acc_d: dict, classes: list):
    print(f"Accuracy Chart")
    print(f"--------------")
    for key, value in acc_d.items():
        print(f"{classes[int(key)]} : {(value * 100):.2f} %")
