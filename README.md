# Глоссирование нивхского языка

Валидация

| Модель      | Attention  | LSTM-слой  |  BPE-слой  | Accuracy   | Word-Level Accuacy |  Epoch  |
|-------------|------------|------------|------------|------------|--------------------|---------|
|  LSTM       |     –      |      -     |    –       |   0.9632   |       0.8314       |  36     |
|  LSTM       |     +      |      -     |    –       |   -        |       0.8347       |  117    |
|  LSTM       |     -      |      -     |    +       |   0.9569   |       0.7983       |  80     |
|  LSTM       |     +      |      -     |    +       |   0.9602   |       0.8116       |  78     |
|  CNN        |     -      |      -     |    -       |   0.9722   |       0.8678       |  75     |
|  CNN        |     +      |      -     |    -       |   0.9700   |       0.8645       |  109    |
|  CNN        |     -      |      +     |    -       |   0.9705   |       0.8562       |  80     |
|  CNN        |     +      |      +     |    -       |   0.9722   |       0.8645       |  89     |
|  CNN        |     -      |      -     |    +       |   -        |       0.8628       |  144    |
|  CNN        |     +      |      -     |    +       |   0.9708   |       0.8628       |  124    |
|  CNN        |     -      |      +     |    +       |   0.9686   |       0.8446       |  34     |
|  CNN        |     +      |      +     |    +       |   0.9675   |       0.8479       |  115    |



Тестовая

| Модель      | Attention  |  LSTM-слой |  BPE-слой  | Accuracy   | Word-Level Accuacy |
|-------------|------------|------------|------------|------------|--------------------|
|LSTM         |      -     |    -       |     -      |  0.9613    |        0.8188      |
|LSTM         |      +     |    -       |     -      |  0.9623    |        0.8159      |
|LSTM         |      -     |    -       |     +      |  0.9444    |        0.7393      |
|LSTM         |      +     |    -       |     +      |  0.9503    |        0.7525      |
|CNN          |      -     |    -       |     -      |  0.9711    |        0.8600      |
|CNN          |      +     |    -       |     -      |  0.9699    |        0.8556      |
|CNN          |      -     |    +       |     -      |  0.9723    |        0.8556      |
|CNN          |      +     |    +       |     -      |  0.9659    |        0.8438      |
|CNN          |      -     |    -       |     +      |  0.9728    |        0.8630      |
|CNN          |      +     |    -       |     +      |  0.9684    |        0.8512      |
|CNN          |      -     |    +       |     +      |  0.9684    |        0.8409      |
|**CNN**          |      +     |    +       |     +      |  0.9713    |        **0.8645**     |

