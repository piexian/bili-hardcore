/// 与协议无关的答题请求，各适配器自行翻译成对应协议的请求体。
/// 思考开关与强度由客户端从配置持有，调用方只负责题目本身。
pub struct QuizRequest<'a> {
    pub question: &'a str,
    pub options: &'a [String],
    pub categories: &'a [String],
}
