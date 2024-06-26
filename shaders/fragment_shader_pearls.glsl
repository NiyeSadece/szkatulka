#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D ourTexture;
uniform vec3 lightPos1;
uniform vec3 lightColor1;
uniform vec3 lightPos2;
uniform vec3 lightColor2;
uniform vec3 viewPos;

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient1 = ambientStrength * lightColor1;
    vec3 ambient2 = ambientStrength * lightColor2;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir1 = normalize(lightPos1 - FragPos);
    vec3 lightDir2 = normalize(lightPos2 - FragPos);
    float diff1 = max(dot(norm, lightDir1), 0.0);
    float diff2 = max(dot(norm, lightDir2), 0.0);
    vec3 diffuse1 = diff1 * lightColor1;
    vec3 diffuse2 = diff2 * lightColor2;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir1 = reflect(-lightDir1, norm);
    vec3 reflectDir2 = reflect(-lightDir2, norm);
    float spec1 = pow(max(dot(viewDir, reflectDir1), 0.0), 32);
    float spec2 = pow(max(dot(viewDir, reflectDir2), 0.0), 32);
    vec3 specular1 = specularStrength * spec1 * lightColor1;
    vec3 specular2 = specularStrength * spec2 * lightColor2;

    // Combine results
    vec3 result = (ambient1 + diffuse1 + specular1 + ambient2 + diffuse2 + specular2) * ourColor;
    FragColor = texture(ourTexture, TexCoord) * vec4(result, 1.0);
}
