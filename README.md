# DC-Lab-1

## Задание 2.
На каждом из графиков представдено время выполнения, ускорение и эффективность алгоритма на различных размерах матриц при разбиении по строкам, по столбцам и по блокам для количества потоков P = 1, 4, 16.

### Графики параллельного выполнения:

<img width="761" height="458" alt="image" src="https://github.com/user-attachments/assets/146eb8de-24d4-4641-b1e6-c1dc1f869ee9" />

<img width="761" height="455" alt="image" src="https://github.com/user-attachments/assets/4492d6a9-3107-417a-a783-0785896dee83" />

<img width="762" height="458" alt="image" src="https://github.com/user-attachments/assets/c0231e47-7fa9-48f1-9b4a-73aa3ff1b03f" />

### Графики последовательного выполнения:

<img width="761" height="454" alt="image" src="https://github.com/user-attachments/assets/bceda8a5-4219-45bf-b837-145c6c03640d" />

<img width="763" height="452" alt="image" src="https://github.com/user-attachments/assets/b1ed1b96-dace-43e4-9467-06693f3be683" />

<img width="758" height="456" alt="image" src="https://github.com/user-attachments/assets/6d2200b5-33fb-4f54-b7d6-bfc8f5b05a96" />



### Графики по ускорению:

<img width="694" height="410" alt="image" src="https://github.com/user-attachments/assets/0ba06896-5924-444b-9c0b-f03be973ef13" />

<img width="697" height="412" alt="image" src="https://github.com/user-attachments/assets/62924570-b03f-4f9f-89b8-af4c4fdbc550" />

<img width="698" height="415" alt="image" src="https://github.com/user-attachments/assets/17e33ba6-4c46-4c4a-92da-2da7f17511cd" />

### Графики по эффективности:

<img width="696" height="410" alt="image" src="https://github.com/user-attachments/assets/3c2d3c2f-ae45-495c-b2fc-ceb496f0f7aa" />

<img width="689" height="413" alt="image" src="https://github.com/user-attachments/assets/6b138476-369f-48e3-8a61-63e98779e058" />

<img width="696" height="414" alt="image" src="https://github.com/user-attachments/assets/d43c8312-fa3a-4912-8a83-8b2e75e8af55" />

### Выводы по заданию 2.
При малых размерах матриц (N <= 64) параллелизация неэффективна из-за того, что накладные расходы на коммуникацию между процессами превышают выигрыш от распределения вычислений. Это классический пример, когда параллелизация становится выгодной только при достаточном объеме вычислительной работы.

При увеличении размера матрицы до N >= 128 наблюдается переходный момент, когда выигрыш от параллелизации начинает превосходить накладные расходы. Для больших матриц (N >= 512) параллелизация становится особенно эффективной, так как объем вычислений растет квадратично (O(N²)), а коммуникационные затраты растут линейно или даже медленнее.

## Задание 3.
На каждом из графиков представлено время при параллельном и последовательном умножении, а также ускорение и эффективность для P = 1, 4, 16

<img width="1039" height="665" alt="image" src="https://github.com/user-attachments/assets/23e2aa21-bf16-435d-ad3a-189c3c7e450d" />

<img width="919" height="663" alt="image" src="https://github.com/user-attachments/assets/f2746ecb-9f8c-4e89-ac85-261334a980b1" />

<img width="1055" height="674" alt="image" src="https://github.com/user-attachments/assets/31cb8c9b-8d1f-4eca-b66d-ea7b1a9fa6cb" />

<img width="1136" height="664" alt="image" src="https://github.com/user-attachments/assets/8aa77910-c3a9-406c-aaf5-04af2a1fddf2" />
