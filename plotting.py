import matplotlib.pyplot as plt


class TrainPlot():
    def __init__(self, title, xlabel, ylabel):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.fig.show()
        self.all_x_train = []
        self.all_y_train = []
        self.all_x_test = []
        self.all_y_test = []
        line1, = self.ax.plot(self.all_x_train, self.all_y_train, 'g')
        line2, = self.ax.plot(self.all_x_test, self.all_y_test, 'b')
        self.ax.legend((line1, line2), ('Train', 'Test'))

    def update(self, x, y, line):
        if line == 'train':
            self.all_x_train.append(x)
            self.all_y_train.append(y)
            self.ax.plot(self.all_x_train, self.all_y_train, 'g')
        else:
            self.all_x_test.append(x)
            self.all_y_test.append(y)
            self.ax.plot(self.all_x_test, self.all_y_test, 'b')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)

    def show(self):
        plt.show()

    def save(self, path):
        self.fig.savefig(path)


class ValuationPlot():
    def __init__(self, title, xlabel, ylabel, y_pred, y_real, only_residual=True):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Sample Index')
        self.fig.show()
        if not only_residual:
            y_pred = y_pred[:50]
            y_real = y_real[:50]
            preds, = self.ax.plot(y_pred, 'go', markersize=6)
            reals, = self.ax.plot(y_real, 'bo', markersize=6)
            zipped = zip(y_pred, y_real)
            for i, (pred, real) in enumerate(zipped):
                self.ax.plot([i, i], [pred, real], 'k-')
            self.ax.legend((preds, reals), ('Predictions', 'Real Values'))
            self.ax.set_ylabel('PM2.5')
        else:
            ress = []
            for i in range(len(y_pred)):
                ress.append(y_pred[i] - y_real[i])
            residuals, = self.ax.plot(ress, 'r.')
            self.ax.legend((residuals,), ('Residuals',))
            self.ax.set_ylabel('PM2.5 Residual')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def close(self):
        plt.close(self.fig)
