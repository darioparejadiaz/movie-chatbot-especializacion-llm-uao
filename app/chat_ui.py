import gradio as gr


class ChatUI:

    # ***************************************

    def __init__(self, llm, search_fn, genres):
        self.__llm = llm
        self.__search_fn = search_fn
        self.__init_UI(genres)

    # ***************************************

    def __init_UI(self, genres):
        with gr.Blocks() as iface:
            gr.Markdown("# Recomendaciones de películas con MovieChatbot")
            gr.Markdown(
                "Escribe la trama de la película que deseas ver, también escoge en los selectores el tipo de género que quieres, una puntuación (1-10) y el número de resultados que deseas."
            )

            query_textbox = gr.Textbox(
                lines=5, placeholder="Escribe tu búsqueda...", label="Búsqueda"
            )
            genre_dropdown = gr.Dropdown(choices=genres, label="Género")
            rating_slider = gr.Slider(
                minimum=1, maximum=10, value=5, label="Puntaje mínimo"
            )
            top_k_number = gr.Number(
                minimum=1, maximum=10, value=3, label="Número de resultados"
            )

            chatbot_response = gr.Textbox(label="Respuesta del Chatbot")

            submit_button = gr.Button("Enviar", elem_id="submit_button")
            clear_button = gr.Button("Limpiar", elem_id="clear_button")

            submit_button.interactive = False

            def update_button(query):
                return gr.update(interactive=bool(query))

            def clear_form():
                return (
                    gr.update(value=""),
                    gr.update(value="Action"),
                    gr.update(value=5),
                    gr.update(value=3),
                    gr.update(interactive=False),
                    gr.update(value=""),
                )

            query_textbox.change(
                fn=update_button, inputs=query_textbox, outputs=submit_button
            )

            submit_button.click(
                fn=self.__interact_with_chatbot,
                inputs=[query_textbox, genre_dropdown, rating_slider, top_k_number],
                outputs=chatbot_response,
            )

            clear_button.click(
                fn=clear_form,
                inputs=None,
                outputs=[
                    query_textbox,
                    genre_dropdown,
                    rating_slider,
                    top_k_number,
                    submit_button,
                    chatbot_response,
                ],
            )

        iface.launch()

    # ***************************************

    def __chatbot_response(self, query, genre=None, rating=None, top_k=5):
        try:
            translation = self.__translate_to_english_user_query(query)
            search_results = self.__search_fn(query, genre, rating, top_k)
            movie_info = self.__format_results_for_chatbot(search_results)
            prompt = f"User asked: '{translation}'. Here are the movie recommendations:\n\n{movie_info}\n\nNow, respond with a helpful and natural explanation to the user in spanish."

            response = self.__llm.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides movie recommendations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            generated_text = response.choices[0].message["content"].strip()

            return generated_text

        except Exception as e:
            return f"Error processing the chatbot response: {str(e)}"

    # ***************************************

    def __translate_to_english_user_query(self, query):
        response = self.__llm.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful spanish to english translator",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )

        generated_text = response.choices[0].message["content"].strip()

        return generated_text

    # ***************************************

    def __interact_with_chatbot(self, query, genre, rating, top_k):
        response = self.__chatbot_response(query, genre, rating, top_k)
        return response

    # ***************************************

    def __format_results_for_chatbot(self, search_results):
        movie_info = ""
        for idx, row in search_results.iterrows():
            movie_info += f"Title: {row['Title']}, Rating: {row['Rating']}, Genre: {row['Genre']}, Year: {row['year']}\n"
            if idx >= 4:
                break
        return movie_info
