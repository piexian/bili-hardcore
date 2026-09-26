/// 把题干与选项拼成对话模型用的纯文本
pub fn build_chat_prompt(question: &str, options: &[String]) -> String {
    format!("题目:{}\n答案:{:?}", question, options)
}

fn build_system_prompt(categories: &[String]) -> String {
    let cat_str = if categories.is_empty() {
        "未知".to_string()
    } else {
        categories.join("、")
    };
    format!(
        "你是一个资深B站用户，目前正在完成硬核会员试炼考试，考试内容涉及的分区：[{}]，面对选择题时，直接根据问题和选项判断正确答案，并返回对应选项的序号（1, 2, 3, 4）。示例：\n\
         问题：大的反义词是什么？\n\
         选项：['长', '宽', '小', '热']\n\
         回答：3\n\
         如果不确定正确答案，选择最接近的选项序号返回，不提供额外解释或超出 1-4 的内容。",
        cat_str
    )
}

fn build_question_text(question: &str, enable_thinking: bool) -> String {
    if enable_thinking {
        question.to_string()
    } else {
        format!("不要思考，直接回答我的问题：{question}")
    }
}

/// 构建 LLM prompt（单条 user 消息，供 Chat Completions 一类协议直接使用）
pub fn build_quiz_prompt(categories: &[String], question: &str, enable_thinking: bool) -> String {
    format!(
        "{}\n---\n{}",
        build_system_prompt(categories),
        build_question_text(question, enable_thinking)
    )
}

/// Claude / Gemini 把人设放独立 system 字段、题目放 user 消息，这里返回两者。
pub fn build_split_prompt(
    categories: &[String],
    question: &str,
    enable_thinking: bool,
) -> (String, String) {
    (
        build_system_prompt(categories),
        build_question_text(question, enable_thinking),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_prompt_keeps_single_user_message_shape() {
        let prompt = build_quiz_prompt(&["科技".to_string()], "题目:1+1\n答案:[1,2]", false);
        assert!(prompt.contains("[科技]"));
        assert!(prompt.ends_with("\n---\n不要思考，直接回答我的问题：题目:1+1\n答案:[1,2]"));

        let thinking = build_quiz_prompt(&[], "题目:1+1", true);
        assert!(thinking.ends_with("\n---\n题目:1+1"));
        assert!(thinking.contains("[未知]"));
    }

    #[test]
    fn split_prompt_separates_system_and_question() {
        let (system, user) = build_split_prompt(&["生活".to_string()], "题目:1+1", true);
        assert!(system.contains("[生活]"));
        assert!(system.contains("你是一个资深B站用户"));
        assert_eq!(user, "题目:1+1");

        let (_, user) = build_split_prompt(&[], "题目:1+1", false);
        assert_eq!(user, "不要思考，直接回答我的问题：题目:1+1");
    }
}
