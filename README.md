# 神经网络梯段下降与反向传播算法学习


## 反向传播原理

  delta_w 怎么设置的

```
  假设loss = sigma((w*x - y)^2)
  为了最小化loss， 我们每次都往loss的下降方向移动delta_loss
  即delta_loss = d(loss)/d(w) * delta_w < 0
  我们将delta_w = - lr * d(loss)/d(w) lr为学习速率一般设置为大于0的数，如0.001
  即： delta_loss = -lr * (d(loss)/d(w))^2
  这样可以保证delta_loss < 0
```

  反向传播推到

```
求导法则: f(g(x)) = f'(g(x)) * g'(x)
记： d(loss)/d(l_1) = delta_l1
l_2 = activate(w * l_1)
d(l_2)/d(l_1) = w*activate_prime(w * l_1)
delta_l1 = d(loss)/d(l_1) = d(loss)/d(l_2)*d(l_2)/d(l_1)
         = delta_l2 * d(l_2)/d(l_1)
         = delta_l2 * w * activate_prime(w * l_1)
d(l_2)/d(w) = activate_prime(w * l_1) * l_1
delta_w = -lr * d(loss)/d(w) = -lr * d(loss)/d(l_2)*d(l_2)/d(w)
        = -lr * delta_l2 * d(l_2)/d(w)
        = -lr * delta_l2 * activate_prime(w * l_1) * l_1
```