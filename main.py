import os

import pandas as pd
import matplotlib.pyplot as plt


class DataVisualizer:
    def __init__(self, json_file_name: str):
        """
                Создает экземпляр класса DataVisualizer на основе данных из JSON файла.

                :param json_file_name: Имя JSON файла с данными.
        """
        self._data_frame = pd.read_json(json_file_name)
        self._total_rooms = len(self._data_frame)

    @staticmethod
    def _get_absolute_path(file_name: str) -> str:
        """
                Возвращает абсолютный путь к файлу сохранения графика.

                :param file_name: Имя файла.
                :return: Абсолютный путь к файлу сохранения графика.
        """
        save_path = f"plots/{file_name}"
        plt.savefig(save_path)
        return os.path.abspath(save_path)

    def corners_prediction_scatter(self, file_name: str = 'corners_prediction_scatter.png',
                                   figure_height: int | float = 8, figure_width: int | float = 8,
                                   matching_color: str = 'green', mismatching_color: str = 'red') -> str:
        """
                Создает график рассеяния для предсказанных и реальных углов комнат.

                :param file_name: Имя файла для сохранения графика.
                :param figure_height: Высота графика.
                :param figure_width: Ширина графика.
                :param matching_color: Цвет точек, где кол-во предсказанных углов совпало с кол-вом реальных.
                :param mismatching_color: Цвет точек, где кол-во предсказанных углов не совпало с кол-вом реальных.
                :return: Абсолютный путь к сохраненному файлу.
        """
        df = self._data_frame
        plt.figure(figsize=(figure_width, figure_height))

        matching_corners = df[df['gt_corners'] == df['rb_corners']]
        mismatching_corners = df[df['gt_corners'] != df['rb_corners']]

        plt.scatter(matching_corners['gt_corners'], matching_corners['rb_corners'], color=matching_color,
                    label='Matching Corners')
        plt.scatter(mismatching_corners['gt_corners'], mismatching_corners['rb_corners'], color=mismatching_color,
                    label='Mismatching Corners')

        plt.title('Scatter Plot of Predicted vs True Corners')
        plt.xlabel('True Corners')
        plt.ylabel('Predicted Corners')

        plt.text(0.5, -0.1, f'Total Rooms: {self._total_rooms}\n'
                            f'Matching Corners: {len(matching_corners)} ({len(matching_corners) / self._total_rooms * 100:.2f}%)\n'
                            f'Mismatching Corners: {len(mismatching_corners)} ({len(mismatching_corners) / self._total_rooms * 100:.2f}%)',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10)

        plt.legend()
        plt.grid(True)
        return self._get_absolute_path(file_name)

    def mean_deviation_histogram(self, file_name: str = 'mean_deviation_histogram.png', figure_height: int | float = 8,
                                 figure_width: int | float = 8, plot_colour: str = 'skyblue') -> str:
        """
                Создает гистограмму среднего отклонения.

                :param file_name: Имя файла для сохранения графика.
                :param figure_height: Высота графика.
                :param figure_width: Ширина графика.
                :param plot_colour: Цвет гистограммы.
                :return: Абсолютный путь к сохраненному файлу.
        """
        df = self._data_frame
        plt.figure(figsize=(figure_width, figure_height))
        plt.hist(df['mean'], bins=20, color=plot_colour, alpha=0.7)

        plt.title('Histogram of Mean Deviation')
        plt.xlabel('Mean Deviation (degrees)')
        plt.ylabel('Frequency')

        ticks = plt.yticks()[0]
        ticks_labels = [f"{int(tick / self._total_rooms * 100)}%" for tick in ticks]
        plt.yticks(ticks, ticks_labels)

        max_value = df['mean'].max()
        plt.xticks(range(0, int(max_value) + 1, 10))
        plt.grid(True)
        return self._get_absolute_path(file_name)

    def mean_floor_vs_ceiling_boxplot(self, file_name: str = 'mean_floor_vs_ceiling_boxplot.png') -> str:
        """
                Создает 2 ящика с усами для сравнения распределения средних значений по полу и потолку.

                :param file_name: Имя файла для сохранения графика.
                :return: Абсолютный путь к сохраненному файлу.
        """
        df = self._data_frame
        fig, ax = plt.subplots()

        ax.boxplot(df['floor_mean'], positions=[1])
        ax.boxplot(df['ceiling_mean'], positions=[2])

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Floor', 'Ceiling'])
        ax.set_xlabel('Distribution')
        ax.set_ylabel('Values')
        plt.title('Comparison of means for Floor and Ceiling')
        return self._get_absolute_path(file_name)
