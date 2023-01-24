from matplotlib import pyplot as plt

def plot_forecast(df, x, y, y_hat, y_upper = None, y_lower = None):
    # input_range = range(df.shape[0])

    x = df[x]
    y = df[y]
    y_hat = df[y_hat]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1, color="black", label="Sign ups")
    plt.plot(x, y_hat, linewidth=1, color="black", label="Predicted sign ups", linestyle='dashed')
    
    if (y_upper is not None)&(y_lower is not None):
        lower_bound = df[y_upper]
        upper_bound = df[y_lower]

        plt.fill_between(
            x,
            lower_bound,
            upper_bound,
            color="blue",
            alpha=0.3,
            label="1SD Confidence Interval",
        )
        
    plt.xlabel("Date")
    plt.ylabel("Sign ups")
    plt.legend()
    plt.show()