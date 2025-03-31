//! Complete example for re-ranking using `gte-multilingual`
fn main() -> gte::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/gte-multilingual-reranker-base/tokenizer.json";
    const MODEL_PATH: &str = "models/gte-multilingual-reranker-base/onnx/model.onnx";
    const APPLY_SIGMOID: bool = true;

    let params = gte::params::Parameters::default().with_sigmoid(APPLY_SIGMOID).with_token_types(true);
    let pipeline = gte::rerank::pipeline::RerankingPipeline::new(TOKENIZER_PATH, &params)?;
    let model = orp::model::Model::new(MODEL_PATH, orp::params::RuntimeParameters::default())?;

    let inputs = gte::rerank::input::TextInput::from_str(&[
        ("Quelle est la capitale de la France ?", "Paris is the capital and largest city of France. With an estimated population of 2,048,472 residents in January 2025 in an area of more than 105 km2 (41 sq mi), Paris is the fourth-most populous city in the European Union and the 30th most densely populated city in the world in 2022. Since the 17th century, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy. Because of its leading role in the arts and sciences and its early adaptation of extensive street lighting, it became known as the City of Light in the 19th century."),
        ("Quelle est la capitale de la France ?", "París es la capital de Francia y su ciudad más poblada. Capital de la región de Isla de Francia (o «Región Parisina»), constituye el único departamento unicomunal del país. Su área metropolitana es la más poblada de toda la Unión Europea."),
        ("Quelle est la capitale de la France ?", "The University of Paris (French: Université de Paris), known metonymically as the Sorbonne, was the leading university in Paris, France, from 1150 to 1970, except for 1793–1806 during the French Revolution. Emerging around 1150 as a corporation associated with the cathedral school of Paris, it was considered the second-oldest university in Europe. Officially chartered in 1200 by King Philip II and recognised in 1215 by Pope Innocent III, it was nicknamed after its theological College of Sorbonne, founded by Robert de Sorbon and chartered by King Louis IX around 1257."),
        ("Quelle est la capitale de la France ?", "L'époque Sengoku (戦国時代, Sengoku-jidai, littéralement « époque des provinces en guerre », en référence à la période des Royaumes combattants chinois) est une ère de l'histoire du Japon marquée par des turbulences sociales, des intrigues politiques et des conflits militaires quasi permanents, qui s'étend du milieu du xve siècle à la fin du xvie siècle au Japon. Au sens strict, cette période débute à la fin des guerres d'Ōnin en 1477 et dure jusqu'en 1573, lorsque le seigneur de la guerre Oda Nobunaga destitue le dernier shogun Ashikaga. Elle couvre approximativement la seconde moitié de l'époque de Muromachi entendue au sens large, qui correspond au shogunat des Ashikaga et s'étend de 1336 à 1573."),
    ]);

    let outputs = model.inference(inputs, &pipeline, &params)?;

    println!("Scores: {:?}", outputs.scores);    

    Ok(())
}
