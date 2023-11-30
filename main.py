from collections import defaultdict
import pandas as pd
from process_img import get_answers_from_exam
import os

def create_defaultdict_from_excel(file_path):
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(file_path)

    # Tạo defaultdict
    answer_dict = defaultdict(list)
    
    # Điền defaultdict với dữ liệu từ DataFrame
    for index, row in df.iterrows():
        question_number = row['Câu ']  # Thay 'Câu ' bằng tên chính xác của cột câu hỏi
        answer = row['Đáp án']
        answer_dict[question_number].append(answer)

    return answer_dict


answer_file_path = 'Answer.xlsx'

excel_defaultdict = create_defaultdict_from_excel(answer_file_path)


result_df = pd.DataFrame(columns=['student', 'score'])
exam_folder = 'student'

data = {
    'student': [],  # Giá trị của cột 1
    'score': []  # Giá trị của cột 2
}

for image_exam in os.listdir('student'):
    image_path = os.path.join(exam_folder, image_exam)

    exam_defaultdict = get_answers_from_exam(image_path)

    score = 0

    for question_number in excel_defaultdict:
        if set(excel_defaultdict[question_number]) == set(exam_defaultdict[question_number]):
            score += 0.25
        
    score = round(score, 3)
    student_name = os.path.splitext(image_exam)[0].split('.')[-1]

    data['student'].append(student_name)
    data['score'].append(score)

# result_df.to_excel('output_file.xlsx', index=False)

df = pd.DataFrame(data)
df.to_excel('score.xlsx', index=False)