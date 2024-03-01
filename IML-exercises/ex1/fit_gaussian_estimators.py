from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    obj = UnivariateGaussian()
    sample = np.random.normal(10, 1, 1000)
    q1 = obj.fit(sample)
    print("(", q1.mu_, ", ", q1.var_, ")")
    # raise NotImplementedError()

    # Question 2 - Empirically showing sample mean is consistent
    res = [0] * 100
    for i in range(100):
        res[i] = obj.fit(sample[:(i + 1) * 10]).mu_
        res[i] = np.abs(10 - res[i])

    # for i in range(len(res)):
    go.Figure([go.Scatter(x=list(range(len(res))), y=res, mode='markers',
                          name=r'$\widehat\mu$')],
              layout=dict(
                  title="Deviation of Sample Mean Estimation",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$text{Sample Mean Estimator }\hat\mu$",
                  height=300)).show()

    # raise NotImplementedError()
    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = q1.pdf(sample)
    go.Figure(go.Scatter(x=sample[:], y=pdfs[:], mode="markers"),
              layout=dict(title="Empirical PDF of fitted model",
                  xaxis_title=r"$x$",
                          yaxis_title=r"$PDF$")).show()
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    obj = MultivariateGaussian()
    cov =   np.array([[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal([0, 0, 4, 0],cov , 1000)
    q1 = obj.fit(samples)
    print(q1.mu_)
    print(q1.cov_)
    # raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    ans = np.zeros((200, 200))
    values = np.linspace(-10, 10, 200)
    for i, f1 in enumerate(values):
        for j, f3 in enumerate(values):
            ans[i, j] = MultivariateGaussian.log_likelihood(
                np.array([f1, 0, f3, 0]), cov, samples)

    go.Figure(go.Heatmap(x=values, y=values, z=ans),
              layout=dict(template="simple_white",
                          title="Log-Likelihood of Multivariate Gaussian Distribution",
                          xaxis_title=r"$\f3$",yaxis_title=r"$\f1$")).show()
    # raise NotImplementedError()

    # Question 6 - Maximum likelihood
    max_index = np.unravel_index(ans.argmax(), ans.shape)

    max_value = [values[max_index[0]], values[max_index[1]]]
    max_value[0] = round(max_value[0],3)
    max_value[1] = round(max_value[1], 3)
################
    print(tuple(max_value))
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
