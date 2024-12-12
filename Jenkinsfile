pipeline {
    agent any
    
    environment {
        // Define environment variables for accessing services
        DOCKER_NETWORK = 'infra_network'
        FLASK_APP_1 = 'flask_app_1'
        FLASK_APP_2 = 'flask_app_2'
        MLFLOW_URI = 'http://172.18.0.1:5003'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Data Ingestion') {
            steps {
                script {
                    // Run data ingestion in one of the Flask containers
                    sh '''
                        docker exec ${FLASK_APP_1} python src/data_ingestion/youtube_comments/main.py
                    '''
                }
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                script {
                    // Run data cleaning and preprocessing
                    sh '''
                        docker exec ${FLASK_APP_1} python src/scripts/main.py
                    '''
                }
            }
        }
        
        stage('Version Data with DVC') {
            steps {
                script {
                    // Use Docker exec to run DVC commands
                    sh '''
                        docker exec ${FLASK_APP_1} dvc add data/train.csv
                        docker exec ${FLASK_APP_1} git add data/train.csv.dvc
                        docker exec ${FLASK_APP_1} git commit -m "Update training data"
                        docker exec ${FLASK_APP_1} dvc push
                    '''
                }
            }
        }
        
        stage('Train Model') {
            steps {
                script {
                    // Run model training script
                    sh '''
                        docker exec ${FLASK_APP_1} python src/models/train_model.py
                    '''
                }
            }
        }
        
        stage('MLflow Tracking') {
            steps {
                script {
                    // Log model to MLflow 
                    sh '''
                        docker exec ${FLASK_APP_1} python src/models/log_model.py
                    '''
                }
            }
        }
        
        stage('Deploy Model') {
            steps {
                script {
                    // Update model in both Flask containers
                    sh '''
                        docker cp ${FLASK_APP_1}:/app/models/latest_model.pkl ./latest_model.pkl
                        docker cp ./latest_model.pkl ${FLASK_APP_2}:/app/models/latest_model.pkl
                    '''
                }
            }
        }
    }
    
    post {
        always {
            // Cleanup steps
            sh 'docker system prune -f'
        }
        
        success {
            // Notification or additional actions on success
            echo 'Pipeline completed successfully!'
        }
        
        failure {
            // Error handling and notifications
            emailext (
                subject: "Jenkins Pipeline Failed: ${currentBuild.fullDisplayName}",
                body: "Pipeline failed. Check console output at ${env.BUILD_URL}",
                recipientProviders: [[$class: 'DevelopersRecipientProvider']]
            )
        }
    }
}