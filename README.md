# Краткое описание эксперимента 

Исследуются паттерны засухи у растений пшеницы. Пшеница выращивается на протяжении нескольких дней в 3-х коробках, содержащих в себе горшки (в каждой коробке по 30 горшков). Через несколько дней она была разделена на 2 группы, одну из которых перестали поливать. В течение 25-ти дней фиксировалось состояние обоих групп (температура, сухой вес), а также были сняты RGB, гиперспектральные (HSI) и температурные (TIR) снимки. Из полученных изображений был сформирован датасет для дальнейшего анализа свойств растения в условиях засухи. 

# Описание задач

1. Совмещение изображений. Несмотря на то, что HSI и TIR снимки были сняты с одной и той же позиции, камеры были разными, поэтому они имеют разные разрешения, а также присутствует отличие в угле съемки и смещении кадра. Для данной задачи был реализован программный интерфейс, позволяющий путем выбора точек совмещения в интерактивном режиме наложить изображение TIR на HSI. Код программы находится в репозитории: https://github.com/Harmonization/Applying-TIR2HSI.

2. Анализ средних спектров. Сигнатурой (или спектром) гиперспектрального изображения (HSI) является его пиксель, имеющий (в данном случае) 204 измерения. Спектральные сигнатуры можно представить как отношение значения канала HSI к его номеру (или длине в нанометрах), тогда график сигнатуры будет иметь уникальную форму, различную для разных материалов. Форма сигнатуры для засушливого и влажного растений отличается, поэтому имеет смысл анализ сигнатур для разных областей изображения. Для анализа средних спектров (сигнатур) в выбранной области изображения (ROI) был реализован программный интерфейс, доступный по ссылке: https://github.com/Harmonization/Analysis-of-HSI-signatures, в котором также рассматриваются специальные матрицы, предназначенные для выявления наиболее эффективного спектрального индекса. 

3. Предобработка и сегментация. Для исследования пшеницы необходимо отделить ее от фона, который включает в себя почву, горшки, колбы с водой и прочие элементы. В случае гиперспектральных изображений эффективным и дешевым методом является вычисление спектрального вегетационного индекса (ВИ), который представляет собой одноканальное изображение, полученное из разности выбранных каналов HSI. Используя один из популярных индексов (или же построив свой), можно отделить растение от фона путем построения маски по пороговому значению над выбранным ВИ. Но изображение растения с отделенным фоном не является окончательным, необходимо также убрать шумовые пиксели (в данных HSI часто встречается шум в последних каналах). Но более важной задачей является удаление аномалий с тепловых изображений, таких как повышенная температура между горшками, из-за чего листья пшеницы находящиеся в этой области имеют некорректно высокую температуру (что является паттерном засухи). Для удаления таких "аномальных" пикселей строится модель линейной регрессии между температурой и одним из каналов HSI (например с наиболее коррелированным), после чего выбираются пороги a и b, которые будут являться допустимой границей удаления пикселей от линии регрессии. Таким образом удаляются пиксели растения между горшками (а также прочие шумовые пиксели и выбросы), которые сильно мешали бы задаче обнаружения засухи. Для данной задачи был написан программный интерфейс, код которого приведен в файле Pixel_Selection_Editor.py. Ссылка на исполняемый файл: https://disk.yandex.ru/d/3dKOXQk1LqS8kg.

4. Классификация дня засухи. Вегетативные спектральные индексы (ВИ) имеют хорошую предсказательную способность для обнаружения засухи у растений. Для данной задачи было выбрано несколько популярных ВИ (а также TIR-снимки) и исследована их эффективность. Для вычисленных индексов были собраны статистические признаки (mean, std, max и пр.). Была построена модель однослойного пересптрона (SLP), которая предсказывала номер дня без полива по стат. признакам некоторых индексов. Был проведен анализ наилучших для данной задачи ВИ и стат. признаков. Данный подход является "дешевым" за счет использования высокоэффективных признаков вместо целого изображения и простой предсказывающей модели. Код для решения данной задачи находится в блокноте Features_Training.ipynb.

5. Предсказание температуры растения. Засыхающее растение имеет более высокую температуру чем влажное (уже на 5-й день заметно повышение температуры у группы без полива), поэтому термальные снимки (TIR) являются эффективными для обнаружения засухи на раннем сроке, когда растение еще можно спасти. Но TIR-снимки сильно зависят от температуры окружающей среды, поэтому в жаркий солнечный день будут совершенно бесполезны (также возможно наличие аномалий, описанных в пункте 3). Поэтому более надежным является применение HSI, которые лишены подобных недостатков. TIR и HSI были совмещены, поэтому каждому пикселю TIR (температура) ставился в соответствие пиксель HSI (сигнатура). Была построена простая модель SLP, позволяющая предсказывать температурные метки по спектральным сигнатурам, а также исследованы наиболее подходящие для этой задачи каналы HSI (удалось снизить с 204 каналов до 8 и меньше).



