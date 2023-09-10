import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Bring AI to your data store',
    description: (
      <>
        Easily deploy, train and manage any AI models and APIs on your datastore: 
        from LLMs, public APIs to highly custom machine learning models, 
        use-cases and workflows.
      </>
    ),
  },
  {
    title: 'Build AI applications on top of your datastore',
    description: (
      <>
        A single scalable deployment of all your AI models and APIs which is 
        automatically kept up-to-date as new data is processed immediately.
      </>
    ),
  },
  {
    title: 'Work with any ML/AI frameworks and APIs',
    description: (
      <>
        Integrate and combine models from Sklearn, PyTorch, HuggingFace
        with AI APIs such as OpenAI to build even the most complex AI
        applications and workflows.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      {/* <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div> */}
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features} style={{paddingTop: '5%'}}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
